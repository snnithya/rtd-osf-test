import os
import torch
import numpy as np
import librosa
import torch.nn as nn
import torch.nn.functional as F

#run model on test audio to get lay ratio
def computeLayRatio(audio=None, fs=16e3, source='vocal', mode='test', audioPath=None, gpuID=None, pretrainedModelDir='../pretrained_models', hopDur=0.5, smoothTol=5, ):
    """
    Compute frame-wise lay ratio (surface tempo multiple) estimates using pre-trained models from [1]. There are separate models available for mixture concert audios, as well as source separated vocals and pakhawaj streams. 
    
    Parameters:
    audio: audio signal sampled at 16kHz
    fs: sampling rate, has to be 16 kHz 
    source: one of 'mix', 'voc' or 'pakh' (mixture, vocal or pakhawaj)
    mode: 'eval' to predict on audio from original dataset, 'test' for new audios not in dataset
    audioPath: path to audio file, if signal not provided 
    gpuID: serial id/number of gpu device to use, if available
    pretrainedModelDir: path to pretrained models (provided in this library)
    hopDur: intervals at which lay ratios are to be obtained on the audio
    smoothTol: minimum duration for which a lay ratio value has to be consistently predicted to be retained during the smoothing. If lesser, the prediction is considered erroneous and replaced with neighbouring values.
    
    Returns:
    List of frame-wise lay ratio values
    """

    if (audio is None) and (audioPath is None):
        print('Provide one of audio signal or path to audio')
        raise
        
    #load input audio
    elif audio is None:
        audio,fs = librosa.load(audioPath, sr=fs)

    #use GPU or CPU
    if gpuID is None:
        device = torch.device("cpu")
    else:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = torch.device("cuda:%d"%gpuID)
        else:
            print("no gpu device found; using cpu")
            device = torch.device("cpu")

    #melgram parameters
    winsize_sec = 0.04
    winsize = int(winsize_sec*fs)
    hopsize_sec = 0.02
    hopsize = int(hopsize_sec*fs)
    nfft = int(2**(np.ceil(np.log2(winsize))))

    input_len_sec = 8
    input_len = int(input_len_sec/hopsize_sec)
    input_hop_sec = hopDur
    input_hop = int(input_hop_sec/hopsize_sec)
    input_height = 40

    #minimum section duration for smoothing s.t.m. estimates
    min_sec_dur = smoothTol #in seconds
    min_sec_dur /= input_hop_sec

    #convert to mel-spectrogram
    melgram = librosa.feature.melspectrogram(audio, sr=fs, n_fft=nfft, hop_length=hopsize, win_length=winsize, n_mels=input_height, fmin=20, fmax=8000)
    melgram = 10*np.log10(1e-10+melgram)
    melgram_chunks = makechunks(melgram, input_len, input_hop)

    #load model
    classes_dict = {'voc':[1.,2.,4.,8.],'pakh':[1.,2.,4.,8.,16.],'mix':[1.,2.,4.,8.,16.]}
    model_ids = [0,1,2]
    model={}
    for i in model_ids:
	    model_path=os.path.join(pretrained_model_dir, source, 'saved_model_fold_%d.pt'%i)
	    model[i]=build_model(input_height,input_len,len(classes_dict[source])).float().to(device)
	    model[i].load_state_dict(torch.load(os.path.join(model_path),map_location=device))
	    model[i].eval()

    #load splits data if mode is eval
    if mode=='eval':
	    splits_data = {}
	    for fold in range(3):
		    splits_data[fold] = np.loadtxt('../splits/%s/fold_%s.csv'%(source,fold), dtype=str)

    #predict lay ratio versus time
    stm_vs_time = []
    for i_chunk, chunk in enumerate(melgram_chunks):
	    model_in = (torch.tensor(chunk).unsqueeze(0)).unsqueeze(1).float().to(device)

	    avg_out = []
	    model_out = {}
	    if mode == 'eval':
		    i_fold = get_fold_num(i_chunk, hopsize_sec, splits_data, boundaries)
		    avg_out = (nn.Softmax(1)(model[i_fold].forward(model_in))).detach().numpy()

	    elif mode=='test':
		    for i in model_ids:
			    model_out[i] = (nn.Softmax(1)(model[i].forward(model_in))).detach().numpy()
			    if len(avg_out) == 0:
				    avg_out = model_out[i].copy()
			    else:
				    avg_out += model_out[i]
		    avg_out/=len(model_ids)

	    stm_vs_time.append(np.argmax(avg_out))

    #smooth predictions with a minimum section duration of 5s
    stm_vs_time = smooth_boundaries(stm_vs_time,min_sec_dur)
    
    return stm_vs_time

#class for sf layers
class sfmodule(nn.Module):
        def __init__(self,n_ch_in):
                super(sfmodule, self).__init__()
                n_filters=16
                self.bn1=nn.BatchNorm2d(n_ch_in,track_running_stats=True)
                self.conv1=nn.Conv2d(n_ch_in, n_filters, (1,5), stride=1, padding=(0,2))
                self.elu=nn.ELU()
                self.do=nn.Dropout(p=0.1)

        def forward(self,x):
                y=self.bn1(x)
                y=self.conv1(x)
                y=self.elu(y)
                y=self.do(y)
                return y

#class for multi-filter module
class mfmodule(nn.Module):
        def __init__(self,pool_height,n_ch,kernel_widths,n_filters):
                super(mfmodule, self).__init__()
                self.avgpool1=nn.AvgPool2d((pool_height,1))
                self.bn1=nn.BatchNorm2d(n_ch,track_running_stats=True)

                self.conv1s=nn.ModuleList([])
                for kw in kernel_widths:
                        self.conv1s.append(nn.Conv2d(n_ch, n_filters[0], (1,kw), stride=1, padding=(0,kw//2)))

                self.do=nn.Dropout(0.5)
                self.conv2=nn.Conv2d(n_filters[0]*len(kernel_widths),n_filters[1],(1,1),stride=1)

        def forward(self,x):
                y=self.avgpool1(x)
                y=self.bn1(y)
                z=[]
                for conv1 in self.conv1s:
                        z.append(conv1(y))
                
                #trim last column to keep width=input_len (needed if filter width is even)
                for i in range(len(z)):
                        z[i]=z[i][:,:,:,:-1]

                y=torch.cat(z,dim=1)
                y=self.do(y)
                y=self.conv2(y)
                return y

#class for dense layers
class densemodule(nn.Module):
        def __init__(self,n_ch_in,input_len,input_height,n_classes):
                super(densemodule, self).__init__()
                n_linear1_in=n_ch_in*input_height
                
                self.dense_mod=nn.ModuleList([nn.AvgPool2d((1,input_len)), 
                nn.BatchNorm2d(n_ch_in,track_running_stats=True), 
                nn.Dropout(p=0.5), 
                nn.Flatten(), 
                nn.Linear(n_linear1_in,n_classes)]) 
                
        def forward(self,x):
                for layer in self.dense_mod:
                        x=layer(x)
                return x

#build model by putting together different layer types
def build_model(input_height,input_len,n_classes):
        model=nn.Sequential()
        i_module=0
        
        #add sf layers
        sfmod_ch_sizes=[1,16,16]
        for ch in sfmod_ch_sizes:   
                sfmod_i=sfmodule(ch)   
                model.add_module(str(i_module),sfmod_i)
                i_module+=1

        #add mfmods
        pool_height=5
        kernel_widths=[16,32,64,96]
        ch_in,ch_out=16,16
        mfmod_n_filters=[12,16]
        
        mfmod_i=mfmodule(pool_height,ch_in,kernel_widths,mfmod_n_filters)
        model.add_module(str(i_module),mfmod_i)
        input_height//=pool_height
        i_module+=1

        #add densemod
        ch_in=16
        densemod=densemodule(ch_in,input_len,input_height,n_classes)
        model.add_module(str(i_module),densemod)
        return model

#function to create N-frame overlapping chunks of the full audio spectrogram  
def makechunks(x,duration,hop):
        n_chunks=int(np.floor((x.shape[1]-duration)/hop) + 1)
        y=np.zeros([n_chunks,x.shape[0],duration])
        for i in range(n_chunks):
                y[i]=x[:,i*hop:(i*hop)+duration]
                #normalise
                y[i]=(y[i]-np.min(y[i]))/(np.max(y[i])-np.min(y[i]))
        return y

#function to smooth predicted s.t.m. estimates by constraining minimum section duration
def smooth_boundaries(stmvstime_track,min_dur):
	stmvstime_track_smu=np.copy(stmvstime_track)
	prev_stm=stmvstime_track_smu[0]
	curr_stm_dur=1
	i=1
	while i < len(stmvstime_track_smu):
		if stmvstime_track_smu[i]!=stmvstime_track_smu[i-1]:
			if curr_stm_dur>=min_dur:
				curr_stm_dur=1
				prev_stm=stmvstime_track_smu[i-1]
				
			else:
				#if stmvstime_track_smu[i]==prev_stm:
				stmvstime_track_smu[i-curr_stm_dur:i]=prev_stm
				#else:
				#	prev_stm=stmvstime_track_smu[i-1]
				curr_stm_dur=1
		else: curr_stm_dur+=1
		i+=1
	return stmvstime_track_smu

def get_fold_num(i_chunk, hopsize_sec, splits_data, boundaries):
	i_chunk_second = i_chunk*hopsize_sec
	section_idx = np.where(np.array([bounds[0]<=i_chunk_second<=bounds[1] for bounds in boundaries])==True)[0][0]
	try:
		fold_num = np.where(np.array(['GB_AhirBhrv_Choutal_%d'%section_idx in splits_data[i] for i in range(3)])==True)[0][0]
	except:
		print(i_chunk_second)
		fold_num = 0
	return fold_num


'''
References

[1] Rohit M. A., Vinutha T. P., and Preeti Rao " Structural Segmentation of Dhrupad Vocal Bandish Audio Based on Tempo, "Proceedings of ISMIR, October 2020, Montreal, Canada
'''
