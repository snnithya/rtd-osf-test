from tkinter import E
import warnings
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import is_color_like, to_rgb
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import scipy.signal as sig
from librosa.display import waveplot, specshow
from IPython.display import Audio, Video 
import parselmouth
import math
import soundfile as sf
import ffmpeg
import os
import cv2
from collections import defaultdict
import utils_fmp as fmp
import pdb

#TODO- add audioPath can be None to all docstrings# #Resolved
#set seaborn theme parameters for plots
sns.set_theme(rc={"xtick.bottom" : True, "ytick.left" : False, "xtick.major.size":4, "xtick.minor.size":2, "ytick.major.size":4, "ytick.minor.size":2, "xtick.labelsize": 10, "ytick.labelsize": 10})

# HELPER FUNCTION
def __check_axes(axes):
	"""Check if ``axes`` is an instance of an ``matplotlib.axes.Axes`` object. If not, use ``plt.gca()``.
	
	This function is a modified version from [1]_.

	.. [1] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

	"""
	
	if axes is None:
		axes = plt.gca()
		
	elif not isinstance(axes, matplotlib.axes.Axes):
		raise ValueError(
			"``axes`` must be an instance of matplotlib.axes.Axes. "
			"Found type(axes)={}".format(type(axes))
		)
	return axes

# ANNOTATION FUNCTION
def readCycleAnnotation(cyclePath, numDiv, startTime, duration, timeCol='Time', labelCol='Cycle'):
	'''Function to read cycle annotation and add divisions between markings if required.

	This function reads the timestamps of provided annotations and adds ``numDiv`` 'computed' annotations between each provided annotation.

	Parameters
	----------
		cyclePath	: str, path object or file-like object
			String path, os.PathLike object or file-like object (containing a read() method) pointing to the cycle annotation csv.

			This value is passed to ``pandas.read_csv()``.
			
		numDiv	: int >= 0
			Number of equally spaced time stamps to add between consecutive provided annotations.

		startTime	: float
			Start time of audio being analysed.

		duration	: float
			Duration of the audio to be analysed.

		timeCol	: str
			Column name of timestamps in cycle annotation file.

		labelCol	: str or None
			Column name of labels for the annotations in the annotation file.

			If None, only ``timeCol`` values will be returned.


	Returns
	-------
		provided	: pandas.DataFrame
			Data frame contains the time stamps on annotations. If ``labelCol`` is not None, also returns respective labels of all provided annotations. 

		computed	: list 
			Timestamps of computed annotations (i.e. annotations computed between consecutive provided annotations).

			If ``numDiv`` is 0 or number of provided annotations is 1, an empty list is returned 

		.. note ::
			If there are no provided annotations present during the relevant duration of audio, the function will return (None, None)
	'''
	
	cycle_df = pd.read_csv(cyclePath)
	index_values = cycle_df.loc[(cycle_df[timeCol] >= startTime) & (cycle_df[timeCol] <= startTime + duration)].index.values
	if len(index_values) == 0:
		return None, None
	provided = cycle_df.iloc[max(index_values[0]-1, 0):min(index_values[-1]+2, cycle_df.shape[0])] 	# filter out rows from annotation file that fall within the considered time duration
	provided = provided.loc[:, [timeCol, labelCol]] if labelCol is not None else provided.loc[:, [timeCol]] 	# retain only the time and label columns from the data frame
	# add divisions in the middle
	computed = []
	for ind, val in enumerate(provided[timeCol].values[:-1]):
		computed.extend(np.around(np.linspace(val, provided[timeCol].values[ind+1], num = numDiv, endpoint=False), 2)[1:])
	return provided, computed

# ANNOTATION FUNCTION
def readOnsetAnnotation(onsetPath, startTime, duration, timeCol='Inst', labelCol='Label'):
	'''Function to read onset annotations.

	Reads an onset annotation csv file and returns the timestamps and annotation labels for annotations within a given time duration.

	Parameters
	----------
		onsetPath	: str, path object or file-like object
			String path, os.PathLike object or file-like object (containing a read() method) pointing to the onset annotation csv.

			This value is passed to ``pandas.read_csv()``.
		
		startTime	: float
			Start time of audio being analysed.
		
		duration	: float
			Duration of the audio to be analysed.
		
		timeCol	: str
			Column name of timestamps in onset annotation file.

		labelCol	: str or None
			Column name in the onset file to take onset labels from. 

			If None, will return only values from the column ``timeCol``.

	Returns
	-------
		provided	: pd.DataFrame
			Dataframe with time stamps and labels (only if ``labelCol`` is not None) of the onsets

			If ``labelCol`` is None, it will return only annotation time stamps.

			If no onsets are present in the given time duration, None is returned.
	'''

	onset_df = pd.read_csv(onsetPath)
	if labelCol is None:
		# if labelCol is None, return only timestamps
		return onset_df.loc[(onset_df[timeCol] >= startTime) & (onset_df[timeCol] <= startTime + duration), [timeCol]]
	else:
		provided = onset_df.loc[(onset_df[timeCol] >= startTime) & (onset_df[timeCol] <= startTime + duration), [timeCol, labelCol]]
	return provided if provided.shape[0] > 0 else None  	# return None if no elements are in provided

# ANNOTATION FUNCTION
def drawAnnotation(cyclePath=None, onsetPath=None, onsetTimeKeyword=None, onsetLabelKeyword=None, numDiv=0, startTime=0, duration=None, ax=None, annotLabel=True, cAnnot=['purple'], providedAlpha=0.8, computedAlpha=0.4, yAnnot=0.7, sizeAnnot=10, textColour=['white']):
	'''Draws annotations on ax

	Plots annotation labels on ``ax`` if provided, else creates a new matplotlib.axes.Axes object and adds the labels to that. 	

	Parameters
	----------
		cyclePath	: str, path object or file-like object
			String path, os.PathLike object or file-like object (containing a read() method) pointing to the tala-related annotation csv.

			This value is passed to ``readCycleAnnotation()``

		onsetPath	: str, path object or file-like object
			String path, os.PathLike object or file-like object (containing a read() method) pointing to the non-tala related annotation csv (example: syllable or performance related annotations).
			
			This value is passed to ``readOnsetAnnotation()``.

			These annotations are only considered if cyclePath is None.
		
		onsetTimeKeyword	: list
			List of column names in the onset file to take onset timestamps from.

			Length of the list should be equal to the length of ``onsetLabelKeyword`` and ``c``.

		onsetLabelKeyword	: list or None
			List of column names in the onset file to take annotation labels from. 
			
			If list is provided, labels will be drawn for each column name in the list. The length of the list should be equal to length of ``onsetTimeKeyword`` and ``c``.
			
			If None, no labels will be plotted corresponding to the onsets (indicated by vertical lines).

			If ``annotLabel`` is False, then ``onsetLabelKeyword`` can be None.

		numDiv	: int >= 0, default=0
			Number of equally spaced time stamps to add between consecutive pairs of annotations.

			Used only if ``cyclePath`` is not None. 

		startTime	: float >= 0, default=0
			Starting timestamp from which to analyse the audio.
		
		duration	: float >= 0 or None
			Duration of audio to be analysed.

			If None, it will analyse the entire audio length.

		ax	: matplotlib.axes.Axes or None
			matplotlib.axes.Axes object to plot in.

			If None, will use ``plt.gca()`` to use the current matplotlib.axes.Axes object.

		annotLabel	: bool, default=True
			If True, will print annotation label along with a vertical line at the annotation time stamp

			If False, will just add a vertical line at the annotation time stamp without the label.

		cAnnot	: list (of color values) 	
			Each value is passed as parameter ``c`` to ``plt.axvline()``.

			One color corresponds to one column name in ``onsetTimeKeyword`` and in ``onsetLabelKeyword``. ::

				len(onsetTimeKeyword) == len(onsetLabelKeyword) == len(c)

		providedAlpha	: scalar or None
			Controls opacity of the provided annotation lines drawn. Value must be within the range 0-1, inclusive.

			Passed to ``plt.axvline()`` as the ``alpha`` parameter.

		computedAlpha	: scalar or None
			Controls opacity of the computed annotation lines drawn. Value must be within the range 0-1, inclusive.

			Passed to ``plt.axvline()`` as the ``alpha`` parameter.

		yAnnot	: float
			Float value from 0-1, inclusive. 
			
			Indicates where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.

		sizeAnnot	: int
			Font size for annotated text. Passed as ``fontsize`` parameter to ``matplotlib.axes.Axes.annotate()``.

		textColour	: list
			List of strings for each ``onsetLabelKeyword``.

	Returns
	-------
		ax	: matplotlib.Axes.axis
			axis that has been plotted in

	Raises
	------
		ValueError
			- If the hyperparameters ``onsetTimeKeyword``, ``onsetLabelKeyword``, ``c`` and ``textColour`` are lists and do not have the same length.
		
			- If both ``cyclePath`` and ``onsetPath`` are None.
		
	'''
	provided = [] 	# list of provided time stamps
	computed = [] 	# list of computed time stamps
	if cyclePath is not None:
		if onsetTimeKeyword is None:
			# if onsetTimeKeyword is None, set it to the default value for cyclePath - Time
			onsetTimeKeyword = ['Time']
		if annotLabel:
			# if annotLabel is True and onsetLabelKeyword is None, use default onsetLabelKeyword value for cyclePath
			onsetLabelKeyword = ['Cycle']
		else:
			# if annotLabel is False, ensure that onsetLabelKeyword and textColor is None
			onsetLabelKeyword = [None for _ in range(len(onsetTimeKeyword))]
			textColour = [None for _ in range(len(onsetTimeKeyword))]
		
		# check that the lengths of onsetTimeKeyword, onsetLabelKeyword, c and textColour are the same
		if not(len(onsetTimeKeyword) == len(onsetLabelKeyword) and len(onsetTimeKeyword) == len(cAnnot) and len(onsetTimeKeyword) == len(textColour)):
			raise ValueError('Please check parameters onsetTimeKeyword, onsetLabelKeyword, c and textColour. If not None, they should be lists of the same length.')
		
		for i in range(len(onsetTimeKeyword)):
			# for each value in onsetTimeKeyword
			temp_provided, temp_computed = readCycleAnnotation(cyclePath=cyclePath, numDiv=numDiv, startTime=startTime, duration=duration, timeCol=onsetTimeKeyword[i], labelCol=onsetLabelKeyword[i])
			
			# append the values to the provided and computed lists
			provided.append(temp_provided)
			computed.append(temp_computed)
		
	elif onsetPath is not None:
		if onsetTimeKeyword is None:
			# if onsetTimeKeyword is None, set it to the default value for onsetPath - Inst
			onsetTimeKeyword = ['Inst']
		if annotLabel:
			# if annotLabel is True and onsetLabelKeyword is None, use default onsetLabelKeyword value for onsetPath
			onsetLabelKeyword = ['Label']
		else:
			# if annotLabel is False, ensure that onsetLabelKeyword and textColor is None
			onsetLabelKeyword = [None for _ in range(len(onsetTimeKeyword))]
			textColour = [None for _ in range(len(onsetTimeKeyword))]
		
		# check that the lengths of onsetTimeKeyword, onsetLabelKeyword, c and textColour are the same
		if not(len(onsetTimeKeyword) == len(onsetLabelKeyword) and len(onsetTimeKeyword) == len(cAnnot) and len(onsetTimeKeyword) == len(textColour)):
			raise ValueError('Please check parameters onsetTimeKeyword, onsetLabelKeyword, cAnnot and textColour. If not None, they should be lists of the same length.')
		
		for i in range(len(onsetTimeKeyword)):
			# for each value in onsetTimeKeyword
			temp_provided = readOnsetAnnotation(onsetPath=onsetPath, startTime=startTime, duration=duration, timeCol=onsetTimeKeyword[i], labelCol=onsetLabelKeyword[i])
			
			# append the values to the provided list
			provided.append(temp_provided)
		
	else:
		raise ValueError('A cycle or onset path has to be provided for annotation')

	# check if ax is None and use current ax if so
	ax = __check_axes(ax)
	# pdb.set_trace()
	if computed is not None:
		# plot computed annotations, valid only when ``cyclePath`` is not None
		for ind, computedArray in enumerate(computed):
			if computedArray is not None:
				# check that the array is not None
				for computedVal in computedArray:
					ax.axvline(computedVal, linestyle='--', c=cAnnot[ind], alpha=computedAlpha)
	if provided is not None:
		# plot the annotations from the file
		for i, providedArray in enumerate(provided):
			firstLabel = True   # marker for first line for each value in  onsetLabelKeyword being plotted; to prevent duplicates from occuring in the legend
			legendLabel = onsetLabelKeyword[i] if onsetLabelKeyword[i] is not None else onsetTimeKeyword[i] 	# if onsetLabelKeyword is None set the label as onsteTimeKeyword
			if providedArray is not None:
				# check that the array is not None
				for _, providedVal in providedArray.iterrows():
					ax.axvline((providedVal[onsetTimeKeyword[i]]), linestyle='-', c=cAnnot[i], label=legendLabel if firstLabel and cyclePath is None else '', alpha=providedAlpha)  # add label only for first line of onset for each keyword
					if firstLabel:  firstLabel = False 	# make firstLabel False after plotting the first line for each value in onsetLabelKeyword
					if annotLabel:
						ylims = ax.get_ylim()   # used to set label at a height defined by ``y``.
						if isinstance(providedVal[onsetLabelKeyword[i]], str):
							ax.annotate(f"{providedVal[onsetLabelKeyword[i]]}", (providedVal[onsetTimeKeyword[i]], (ylims[1]-ylims[0])*yAnnot + ylims[0]), bbox=dict(facecolor='grey', edgecolor='white'), c=textColour[i], fontsize=sizeAnnot)
						else:
							ax.annotate(f"{float(providedVal[onsetLabelKeyword[i]]):g}", (providedVal[onsetTimeKeyword[i]], (ylims[1]-ylims[0])*yAnnot + ylims[0]), bbox=dict(facecolor='grey', edgecolor='white'), c=textColour[i], fontsize=sizeAnnot)
	if onsetPath is not None and cyclePath is None and len(onsetTimeKeyword) > 1:     # add legend only if multiple onsets are given
		ax.legend()
	return ax

# COMPUTATION FUNCTION
def pitchContour(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, minPitch=98, maxPitch=660, tonic=None, timeStep=0.01, octaveJumpCost=0.9, veryAccurate=True, ax=None, **kwargs): 
	'''Returns pitch contour (in cents) for the audio

	Calculates the pitch contour of a given audio sample using autocorrelation method described in [1]_. The implementation of the algorithm is done using [2]_ and it's Python API [3]_. The pitch contour is converted to cents by making the tonic correspond to 0 cents.

	.. [1] Paul Boersma (1993): "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound." Proceedings of the Institute of Phonetic Sciences 17: 97–110. University of Amsterdam.Available on http://www.fon.hum.uva.nl/paul/

	.. [2] Boersma, P., & Weenink, D. (2021). Praat: doing phonetics by computer [Computer program]. Version 6.1.38, retrieved 2 January 2021 from http://www.praat.org/

	.. [3] Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

	..note ::
		The audio signal is given in mono format to the pitch detection algorithm.

	Uses ``plotPitch()`` to plot pitch contour if ``ax`` is not None.

	Parameters
	----------
		audio	: ndarray or None
			Loaded audio time series

			Audio signal is converted to mono to compute the pitch.

			If None, ``audioPath`` can not be None

		sr	: number > 0; default=16000
			If audio is not None, defines sample rate of audio time series.

			If audio is None and audioPath is not None, defines sample rate to load the audio at.

		audioPath	: str, int, pathlib.Path, file-like object or None
			Path to the input file.

			Used only if audio is None. Audio is loaded as mono.

			Sent to ``librosa.load()`` as ``path``. 

			If None, ``audio`` cannot be None.

		startTime	: float; default=0
			Time stamp to consider audio from.

		duration	: float or None; default=None
			Duration of the audio to consider.

			If duration is None
				- If ``audio`` is None, duration is inferred from the audio.
				- If ``audio`` is None and ``audioPath`` is not None, the entire song is loaded.

		minPitch	: float; default=98
			Minimum pitch (in Hz) to read for contour extraction.

			Passed as ``pitch_floor`` parameter to ``parselmouth.Sound.to_pitch_ac()``.

		maxPitch	: float; default=660
			Maximum pitch to read for contour extraction.

			Passed as ``pitch_ceil`` parameter to ``parselmouth.Sound.to_pitch_ac()``.

		tonic	: float or None
			Tonic of the audio (in Hz). Used to compute the pitch contour in cents. If float is given, returns pitch contour values in cents.

			If None, returns pitch contour in Hz.

		timeStep	: float; default=0.01
			Time steps (in seconds) in which pitch values are extracted. ::

				Example: timeStep = 0.01 implies that pitch values are extracted for every 0.01 s.

		octaveJumpCost	: float
			Degree of disfavouring of pitch changes, relative to maximum possible autocorrelation.

			Passed as ``octave_jump_cost`` parameter to ``praat.Sound.to_pitch_ac()``.

		veryAccurate	: bool
			Determines the type and length of the window used in the pitch extraction algorithm.

			Passed as ``very_accurate`` parameter to ``praat.Sound.to_pitch_ac()``.

		ax	: matplotlib.axes.Axes or None
			Axes to plot the pitch contour in.

		kwargs	: Additional arguements to ``plotPitch()``.

	Returns
	-------
		ax : matplotlib.axes.Axes
			Plot of pitch contour if ``ax`` was not None

		(p, t)	: (ndarray, ndarray)
			Tuple with arrays of pitch values (in cents) and time stamps. Returned if ax was None.
	'''
	
	if audio is None:
		# if audio is not given, load audio from audioPath
		audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
	else:
		# if audio is provided, check that it is in mono and convert it to mono if it isn't
		audio = librosa.to_mono(audio)

	if duration is None:
		duration = librosa.get_duration(audio, sr=sr)

	snd = parselmouth.Sound(audio, sr)
	pitch = snd.to_pitch_ac(time_step=timeStep, pitch_floor=minPitch, very_accurate=veryAccurate, octave_jump_cost=octaveJumpCost, pitch_ceiling=maxPitch) 	# extracting pitch contour (in Hz)

	p = pitch.selected_array['frequency']
	p[p==0] = np.nan    # mark unvoiced regions as np.nan
	if tonic is not None:   
		p[~(np.isnan(p))] = 1200*np.log2(p[~(np.isnan(p))]/tonic)    # convert Hz to cents
		is_cents = True 	# boolean is True if p is in cents
	else:
		is_cents = False 	# boolean is False if p is in Hz
	t = pitch.xs() + startTime
	if ax is None:
		return (p, t)
	else:
		# plot the contour
		return plotPitch(p, t, is_cents=is_cents, ax=ax, **kwargs)

# Nithya: AskRohit - I have made the default values of xticks and xlabel as False and the defaults of yticks and ylabel as True.
# PLOTTING FUNCTION
def plotPitch(p=None, t=None, is_cents=False, notes=None, ax=None, freqXlabels=5, xticks=False, yticks=True, xlabel=False, ylabel=True, title='Pitch Contour', annotate=False, ylim=None, c='blue',**kwargs):
	'''Plots the pitch contour

	Plots the pitch contour passed in the ``p`` parameter, computed from ``pitchContour()``. 

	Parameters
	----------
		p	: ndarray
			Pitch values (in cents).

			Computed from ``pitchContour()``

		t	: ndarray or None
			Time stamps (in seconds) corresponding to each value in ``p``.

			If None, assumes time starts from 0 s with 0.01 s hops for each value in ``p``.

			Computed from ``pitchContour()``.
		
		is_cents	: boolean; default=False
			If True, indicates that ``p`` is in Cents.

			If False, indicates that ``p`` is in Hertz.

		notes	: list or None
			list of dictionaries with keys (``cents`` or ``hertz``) and ``label`` for each note present in the raga of the audio. Uses the ``label`` value as a yticklabel in the plot. ::

				Example:
				notes = [
					{
						"label": "P_",
						"cents": -500
					},
					{
						"label": "D_",
						"cents": -300
					},
					{
						"label": "S",
						"cents": 0
					}
					...
					] 

				If None, uses the cents/Hz values as the yticklabels.

		ax	: matplotlib.axes.Axes or None
			Object on which pitch contour is to be plotted

			If None, will plot in ``matplotlib.pyplot.gca()``.

		freqXlabels	: int
			Time (in seconds) after which each x ticklabel should occur

		xticks	: bool
			If True, will print x ticklabels in the plot.

		yticks	: bool
			If True, will print y ticklabels in the plot.

		xlabel	: bool
			If True, will add label to x axis.

		ylabel	: bool
			If True, will add label to y axis.

		title	: str
			Title to add to the plot.

		annotate	: bool
			If True, will add tala-related/onset annotations to the plot .

		ylim	: (float, float) or None
			(min, max) limits for the y axis.
			
			If None, y limits will be directly interpreted from the data.

		c	: color
			Colour of the pitch contour plotted.

		kwargs	: additional arguements passed to ``drawAnnotation()`` if ``annotate`` is True.

	Returns
	-------
		ax	: matplotlib.axes.Axes
			Plot of pitch contour.

	Raises
	------
		ValueError
			If ``p`` is None.
	'''

	# Check that all required parameters are present
	if p is None:
		ValueError('No pitch contour provided')
	if t is None:
		t = np.arange(0, len(p)*0.01, 0.01)
	#TODO-Rohit: Added below block -> Nithya- I changed duration to take the difference between the last and first time step #Resolved
	
	# if ax is None, use the ``plt.gca()`` to use current axes object
	ax = __check_axes(ax)
	
	ax = sns.lineplot(x=t, y=p, ax=ax, color=c)
	ax.set(xlabel='Time (s)' if xlabel else '', 
	title=title, 
	xlim=(t[0], t[-1]), 
	xticks=np.around(np.arange(math.ceil(t[0]), math.floor(t[-1]), freqXlabels)).astype(int),     # start the xticks such that each one corresponds to an integer with xticklabels
	xticklabels=np.around(np.arange(math.ceil(t[0]), math.floor(t[-1]), freqXlabels)).astype(int) if xticks else []) 	# let the labels start from the integer values.

	# set ylabel according to ``is_cents`` variable
	if is_cents:
		ax.set(ylabel='Pitch (Cents)' if ylabel else '')
	else:
		ax.set(ylabel='Pitch (Hz)' if ylabel else '')
	if notes is not None and yticks:
		# add notes on the yticklabels if notes is not None

		# keyword in the ``notes`` parameter to the get the pitch values of each note from
		if is_cents:
			notes_keyword = 'cents'
		else:
			notes_keyword = 'hertz'
		ax.set(
		yticks=[x[notes_keyword] for x in notes if (x[notes_keyword] >= min(p[~(np.isnan(p))])) & (x[notes_keyword] <= max(p[~(np.isnan(p))]))] if yticks else [], 
		yticklabels=[x['label'] for x in notes if (x[notes_keyword] >= min(p[~(np.isnan(p))])) & (x[notes_keyword] <= max(p[~(np.isnan(p))]))] if yticks else [])
	if ylim is not None:
		ax.set(ylim=ylim)

	if annotate:
		ax = drawAnnotation(startTime=t[0], duration=t[-1]-t[0], ax=ax, **kwargs)

	return ax

# COMPUTATION FUNCTION
def spectrogram(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, winSize=640, hopSize=160, nFFT=1024, ax=None, amin=1e-5, **kwargs): 
	'''Computes spectrogram from the audio sample

	Returns a plotted spectrogram if ax is not None, else returns the computed STFT on the audio.

	Uses ``librosa.display.specshow()`` to plot the spectrogram.

	Parameters
	----------
		audio	: ndarray or None
			Loaded audio time series

			Audio signal is converted to mono to compute the spectrogram.

			If None, ``audioPath`` can not be None.

		sr	: number > 0; default=16000
			If audio is not None, defines sample rate of audio time series.

			If audio is None and audioPath is not None, defines sample rate to load the audio at.

		audioPath	: str, int, pathlib.Path, file-like object or None
			Path to the input file.

			Used only if audio is None. Audio is loaded as mono.

			Sent to ``librosa.load()`` as ``path`` parameter. 

			If None, ``audio`` cannot be None.

		startTime	: float; default=0
			Time stamp to consider audio from.

		duration	: float or None; default=None
			Duration of the audio to consider.

			If ``duration`` is None
				- If ``audio`` is None, duration is inferred from the audio.
				- If ``audio`` is None and ``audioPath`` is not None, the entire song is loaded.

		winSize    : int > 0
			Size of window for STFT (in frames)

		hopSize    : int > 0
			Size of hop for STFT (in frames)

		nFFT    : int or None
			DFT size

			If nFFT is None, it takes the value of the closest power of 2 >= winSize (in samples).

		ax	: matplotlib.axes.Axes or None
			Axes to plot spectrogram in. 

			If None, returns (sample frequencies, segment times, STFT) of audio sample

		amin	: float > 0
			Minimum threshold for ``abs(S)`` and ``ref`` in ``librosa.power_to_db()``. Controls the contrast of the spectrogram.
			
			Passed into ``librosa.power_to_db()`` function.

		kwargs	: additional arguements passed to ``plotSpectrogram()``.
		
	Returns
	-------
		ax	: matplotlib.axes.Axes or (ndarray, ndarray, ndarray)
			If ``ax`` is not None, returns a plot of the spectrogram computed

			If ``ax`` is None, returns a tuple with (sample frequencies, segment times, STFT of the audio (in dB)) computed by ``scipy.signal.stft()``.
	'''
	
	if audio is None:
		audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
	if duration is None:
		duration = librosa.get_duration(audio, sr=sr)
	
	if nFFT is None:
		nFFT = int(2**np.ceil(np.log2(winSize)))     # set value of ``nFFT`` if it is None.

	# STFT
	f,t,X = sig.stft(audio, fs=sr, window='hann', nperseg=winSize, noverlap=(winSize-hopSize), nfft=nFFT)
	X_dB = librosa.power_to_db(np.abs(X), ref = np.max, amin=amin)
	t += startTime 	# add start time to time segments extracted.

	if ax is None:
		# return f, t, X_dB
		return (f, t, X_dB)

	else:
		return plotSpectrogram(X_dB, t, f, sr=sr, ax=ax, **kwargs)

# PLOTTING FUNCTION
def plotSpectrogram(X_dB, t, f, sr=16000, hopSize=160, cmap='Blues', ax=None, freqXlabels=5, freqYlabels=2000, xticks=False, yticks=True, xlabel=False, ylabel=True, title='Spectrogram', annotate=True, ylim=None, **kwargs): 
	'''Plots spectrogram

	Uses ``librosa.display.specshow()`` to plot a spectrogram from a computed STFT. Annotations can be added is ``annotate`` is True.

	Parameters
	----------
	X_dB	: ndarray
		STFT of audio. Computed in ``spectrogram()``.

	t	: ndarray or None
		Time segments corresponding to ``X_dB``.

		If None, will assign time steps from 0 (in seconds).
	
	f	: ndarray
		Frequency values. Computed in ``spectrogram()``.

		If None, will infer frequency values in a linear scale.

	sr	: number > 0; default=16000
		Sample rate of audio processed in ``spectrogram()``.

	hopSize	: int > 0
		Size of hop for STFT (in frames)

	cmap	: matplotlib.colors.Colormap or str
		Colormap to use to plot spectrogram.

		Sent as a parameter to ``plotSpectrogram``.

	ax	: matplotlib.axes.Axes or None
		Axes to plot spectrogram in. 

		If None, plots the spectrogram returned by ``plt.gca()``.

	freqXlabels	: float > 0
		Time (in seconds) after which each x label occurs in the plot
	
	freqYlabels :	float > 0
		Number of Hz after which y label should appear on the Y axis.

	xticks	: bool
		If True, will add xticklabels to plot.

	yticks	: bool
		If True, will add yticklabels to plot.

	xlabel	: bool
		If True, will print xlabel in the plot.

	ylabel	: bool
		If True will print ylabel in the plot.

	title	: str
		Title to add to the plot.

	annotate	: bool
		If True, will annotate markings in either cyclePath or onsetPath with preference to cyclePath.

	ylim	: (float, float) or None
		(min, max) limits for the y axis.
		
		If None, the range is set to (0, sr//2)

	kwargs	: Additional arguements provided to ``drawAnnotation()`` if ``annotate`` is True.
	
	'''
	 #Resolved # TODO-Rohit: for some reason, below line is throwing an error due to x_coords and y_coords; I'm passing o/ps X,t,f from spectrogram function; if x_coords, y_coords not passed then function plots without error; need to debug
	 # Nithya: I am not getting this error, can you tell me what the error says? 
	specshow(X_dB, x_coords=t, y_coords=f, x_axis='time', y_axis='linear', sr=sr, fmax=sr//2, hop_length=hopSize, ax=ax, cmap=cmap, shading='auto')
	#specshow(X_dB,x_axis='time', y_axis='linear', sr=sr, fmax=sr//2, hop_length=hopSize, ax=ax, cmap=cmap)

	# set ylim if required
	if ylim is None:
		ylim = (0, sr//2)

	# set axes params
	ax.set(ylabel='Frequency (Hz)' if ylabel else '', 
	xlabel='Time (s)' if xlabel else '', 
	title=title,
	xlim=(t[0], t[-1]), 
	xticks=np.around(np.arange(math.ceil(t[0]), math.floor(t[-1]), freqXlabels)).astype(int),     # start the xticks such that each one corresponds to an integer with xticklabels
	xticklabels=np.around(np.arange(math.ceil(t[0]), math.floor(t[-1]), freqXlabels)).astype(int) if xticks else [], 	# let the labels start from the integer values.
	ylim=ylim,
	yticks= np.arange(math.ceil(ylim[0]/1000)*1000, math.ceil(ylim[1]/1000)*1000, freqYlabels) if yticks else [], #TODO-AskRohit: try to see if you can make this more general#
	yticklabels=[f'{(x/1000).astype(int)}k' for x in np.arange(math.ceil(ylim[0]/1000)*1000, math.ceil(ylim[1]/1000)*1000, freqYlabels)]  if yticks else [])

	if annotate:
		ax = drawAnnotation(startTime=t[0], duration=t[-1]-t[0], ax=ax, **kwargs)

	return ax

# PLOTTING FUNCTION
def drawWave(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, ax=None, xticks=False, yticks=True, xlabel=False, ylabel=True, title='Waveform', freqXlabels=5, annotate=False, odf=False, winSize_odf=640, hopSize_odf=160, nFFT_odf=1024, source_odf='vocal', cOdf='black', ylim=None, **kwargs): 
	'''Plots the wave plot of the audio

	Plots the waveform of the given audio using ``librosa.display.waveshow()``.

	Parameters
	----------
		audio	: ndarray or None
			Loaded audio time series

		sr	: number > 0; default=16000
			If audio is not None, defines sample rate of audio time series 

			If audio is None and audioPath is not None, defines sample rate to load the audio at

		audioPath	: str, int, pathlib.Path, file-like object or None
			Path to the input file.

			Used only if audio is None. Audio is loaded as mono.

			Sent to ``librosa.load()`` as ``path`` parameter. 

			If None, ``audio`` cannot be None.

		startTime	: float; default=0
			Time stamp to consider audio from.

		duration	: float or None; default=None
			Duration of the audio to consider.

			If duration is None
				- If ``audio`` is None, duration is inferred from the audio.
				- If ``audio`` is None and ``audioPath`` is not None, the entire song is loaded.

		ax	: matplotlib.axes.Axes or None
			Axes to plot waveplot in.

			If None, will plot the object in ``plt.gca()``

		xticks	: bool
			If True, will add xticklabels to plot.

		yticks	: bool
			If True, will add yticklabels to plot.

		xlabel	: bool
			If True, will print xlabel in the plot.

		ylabel	: bool
			If True will print ylabel in the plot.

		title	: str
			Title to add to the plot.

		freqXlabels	: float > 0
			Time (in seconds) after which each x ticklabel occurs in the plot.

		annotate	: boolean
			If True, will annotate markings in either cyclePath or onsetPath with preference to cyclePath.

		odf	: bool
			If True, will plot the onset detection function over the wave form.

			Uses ``getODF()`` to compute ODF.

		winSize_odf    : int
			Window size (in frames) used by the onset detection function.

			If ``odf`` is True, passed to the ``getODF()`` function.

		hopSize_odf    : int
			Hop size (in frames) used by the onset detection function.

			If ``odf`` is True, passed to the ``getODF()`` function.

		nFFT_odf    : int
			Size of DFT used in onset detection function.

			If ``odf`` is True, passed to the ``getODF()`` function.

		source_odf	: str
			Defines type of instrument in the audio. Accepted values are:
			- 'vocal'
			- 'pakhawaj'
			
			Used in the ``getODF()`` only if ``odf`` is True.

		cOdf	: color 
			Colour to plot onset detection function in.

			If ``odf`` is True, passed to the ``getODF()`` function.

		ylim	: (float, float) or None
			(min, max) limits for the y axis.
			
			If None, will be directly interpreted from the data.

		kwargs	: Additional arguements passed to ``drawAnnotation()`` if ``annotate`` is True.
	
	Returns
	-------
		ax	: matplotlib.axes.Axes
			Waveform plot of given audio.

	'''
	
	if audio is None:
		audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
	if duration is None:
		duration = librosa.get_duration(audio, sr=sr)
	
	waveplot(audio, sr, ax=ax)

	if odf:
		plotODF(audio=audio, sr=sr, startTime=0, duration=None, ax=ax, winSize_odf=winSize_odf, hopSize_odf=hopSize_odf, nFFT_odf=nFFT_odf, source_odf=source_odf, cOdf=cOdf, ylim=True) 	# startTime=0 and duration=None because audio is already loaded.
		#Resolved #TODO-AskRohit: is plotODF required in other functions like pitch contour/spectrogram plots #

	# set ylim if required
	if ylim is None:
		ylim = ax.get_ylim()

	ax.set(xlabel='' if not xlabel else 'Time (s)', 
	ylabel = '' if not ylabel else 'Amplitude',
	xlim=(0, duration), 
	xticks=[] if not xticks else np.around(np.arange(math.ceil(startTime) - startTime, duration, freqXlabels)),
	xticklabels=[] if not xticks else np.around(np.arange(math.ceil(startTime), duration+startTime, freqXlabels)).astype(int),
	yticks=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 1), 
	yticklabels=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 1), 
	ylim=ylim,
	title=title)

	if annotate:
		ax = drawAnnotation(startTime=startTime, duration=duration, ax=ax, **kwargs)
	
	return ax

# PLOTTING FUNCTION
def plotODF(audio=None, sr=16000, audioPath=None, odf=None, startTime=0, duration=None, ax=None, winSize_odf=640, hopSize_odf=160, nFFT_odf=1024, source_odf='vocal', cOdf='black', updatePlot=False, xlabel=False, ylabel=False, xticks=False, yticks=False, title='Onset Detection Function', freqXlabels=5, ylim=True, annotate=False, **kwargs):
	#Resolved #TODO-Rohit: added additional 'odf' parameter; in the spirit of separating our computation and plotting functions, this function should also ideally just plot odf, given odf as a parameter. But for now, I've added odf as a parameter and not removed audio input.
	'''
	Plots onset detection function if ``ax`` is provided. Function comes from ``getODF()``.
	
	If ``ax`` is None, function returns a tuple with 2 arrays - onset detection function values and time stamps

	Parameters
	----------
		audio	: ndarray or None
			Loaded audio time series

		sr	: number > 0; default=16000
			If audio is not None, defines sample rate of audio time series 

			If audio is None and audioPath is not None, defines sample rate to load the audio at

		audioPath	: str, int, pathlib.Path, file-like object or None
			Path to the input file.

			Used only if audio is None. Audio is loaded as mono.

			Sent to ``librosa.load()`` as ``path`` parameter. 

			If None, ``audio`` cannot be None.

		odf : ndarray
			Extracted onset detection function, if already available
			
			Can be obtained using ``getODF()`` function

		startTime	: float; default=0
			Time stamp to consider audio from.

		duration	: float or None; default=None
			Duration of the audio to consider.

			If ``duration`` is None
				- If ``audio`` is None, duration is inferred from the audio.
				- If ``audio`` is None and ``audioPath`` is not None, the entire song is loaded.

		ax	: matplotlib.axes.Axes
			Axes object to plot waveplot in.

		winSize_odf    : int
			Window size (in frames) used by the onset detection function.

			If ``odf`` is True, passed to the ``getODF()`` function.

		hopSize_odf    : int
			Hop size (in frames) used by the onset detection function.

			If ``odf`` is True, passed to the ``getODF()`` function.

		nFFT_odf    : int
			Size of DFT used in onset detection function.

			If ``odf`` is True, passed to the ``getODF()`` function.

		source_odf	: str
			Defines type of instrument in the audio. Accepted values are:
			- 'vocal'
			- 'pakhawaj'
			
			Used in the ``getODF()`` only if ``odf`` is True.

		cOdf	: color 
			Colour to plot onset detection function in.

			If ``odf`` is True, passed to the ``getOnsetActivation()`` function.

		updatePlot  : bool
			If odf being plotting on figure with waveform, then retain axis labels and properties

		xticks	: bool
			If True, will add xticklabels to plot.

		yticks	: bool
			If True, will add yticklabels to plot.

		xlabel	: bool
			If True, will print xlabel in the plot.

		ylabel	: bool
			If True will print ylabel in the plot.

		title	: str
			Title to add to the plot.

		freqXlabels	: float > 0
			Time (in seconds) after which each x ticklabel occurs in the plot.

		ylim	: (float, float) or None
			(min, max) limits for the y axis.
			
			If None, will be directly interpreted from the data.

		annotate	: boolean
			If True, will annotate markings in either cyclePath or onsetPath with preference to cyclePath.

		kwargs	: Additional arguements passed to ``drawAnnotation()`` if ``annotate`` is True.

	Returns
	-------
		ax	: matplotlib.axes.Axes)
			If ``ax`` is not None, returns a plot
		
		(odf_vals, time_vals): (ndarray, ndarray)
			If ``ax`` is None, returns a tuple with ODF values and time stamps.
	'''

	if odf is None:

		if audio is None:
			audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
		if duration is None:
			duration = librosa.get_duration(audio, sr=sr)
		
		odf_vals, _ = getODF(audio=audio, audioPath=None, startTime=startTime, duration=duration, fs=sr, winSize=winSize_odf, hopSize=hopSize_odf, nFFT=nFFT_odf, source=source_odf)

	else:
		odf_vals = odf.copy()
		duration = len(odf_vals)*hopSize_odf/sr
	
	# set time and odf values in variables
	time_vals = np.arange(startTime, startTime+duration, hopSize_odf/sr)
	#Resolved #TODO-Rohit: changed last argument to hopsize_odf/sr because hopsize_odf is in frames now 
	
	#Resolved # TODO-Rohit: not sure above line is necessary. I got length mismatch errors so commented it. We could instead add code to make lengths same like below:
	odf_vals = odf_vals[: min((len(time_vals), len(time_vals)))]
	time_vals = time_vals[: min((len(time_vals), len(time_vals)))]


	if ax is None:
		# if ax is None, return (odf_vals, time_vals)
		return (odf_vals, time_vals)
	else:
		#Resolved #TODO-Rohit added below 'if updatePlot' block to retain existing ax properties in case odf is being plotted on top of waveform. Can add this block in other plotting functions too
		if updatePlot:
			xticks_, yticks_, xticklabels_, yticklabels_, title_, xlabel_, ylabel_ = ax.get_xticks(), ax.get_yticks(), ax.get_xticklabels(), ax.get_yticklabels(), ax.get_title(), ax.get_xlabel(), ax.get_ylabel()
		else:
			xticks_, yticks_, xticklabels_, yticklabels_, title_, xlabel_, ylabel_ = [],[],[],[],title,'',''

		ax.plot(time_vals, odf_vals, c=cOdf)     # plot odf_vals and consider odf_vals for all values except the last frame
		max_abs_val = max(abs(odf_vals))   # find maximum value to set y limits to ensure symmetrical plot
		# set ax parameters only if they are not None
		ax.set(xlabel=xlabel_ if not xlabel else 'Time (s)', 
		ylabel = ylabel_ if not ylabel else 'ODF',
		xlim=(0, duration), 
		xticks=xticks_ if not xticks else np.around(np.arange(math.ceil(startTime), duration+startTime, freqXlabels)),
		xticklabels=xticklabels_ if not xticks else np.around(np.arange(math.ceil(startTime), duration+startTime, freqXlabels)).astype(int),
		yticks=yticks_ if not yticks else np.around(np.linspace(-max_abs_val,max_abs_val, 3), 2), #Resolved #TODO-Rohit linspace args edited from min & max(audio)
		yticklabels=yticklabels_ if not yticks else np.around(np.linspace(-max_abs_val,max_abs_val, 3), 2), #Resolved #TODO-Rohit linspace args edited from min & max(audio) - AskRohit: shouldn't the first and second arguement also be -max_abs_val and max_abs_val (like int the previous line) since these are the corresponding label values for the ticks marked in the previous line.
		ylim= ax.get_ylim() if ylim is not None else (-max_abs_val, max_abs_val),
		title=title_) #Resolved #TODO- AskRohit: should the title parameter be removed from this function if it isn't being used?

		# Added by Nithya to add annotations to the plot
		if annotate:
			ax = drawAnnotation(startTime=startTime, duration=duration, ax=ax, **kwargs)
		return ax

# AUDIO MANIPULATION	
def playAudio(audio=None, sr=16000, audioPath=None, startTime=0, duration=None):
	'''Plays relevant part of audio

	Parameters
	----------
		audio	: ndarray or None
			Loaded audio time series.

		sr	: number > 0; default=16000
			If audio is not None, defines sample rate of audio time series .

			If audio is None and audioPath is not None, defines sample rate to load the audio at.

		audioPath	: str, int, pathlib.Path, file-like object or None
			Path to the input file.

			Used only if audio is None. Audio is loaded as mono.

			Sent to ``librosa.load()`` as ``path`` parameter.

			If None, ``audio`` cannot be None.

		startTime	: float; default=0
			Time stamp to consider audio from.

		duration	: float or None; default=None
			If duration is None
				- If ``audio`` is None, duration is inferred from the audio.
				- If ``audio`` is None and ``audioPath`` is not None, the entire song is loaded.
	Returns
	-------
		iPython.display.Audio 
			Object that plays the audio.
	'''
	if audio is None:
		#Resolved # TODO Nithya: I changed the sr parameter value to sr here (from None). Not sure if I had added the None or not, let me know if you think it should be different
		audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
	return Audio(audio, rate=sr)

# AUDIO MANIPULATION
def playAudioWClicks(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, onsetFile=None, onsetLabels=['Inst', 'Tabla'], destPath=None):
	'''Plays relevant part of audio along with clicks at timestamps of each of the onsetLabels provided.

	If ``destPath`` is not None, generated audio is saved at ``destPath``, else the generated audio is returned as a ``iPython.display.Audio`` object.

	Parameters
	----------
		audio	: ndarray or None
			Loaded audio time series.

		sr	: number > 0; default=16000
			If audio is not None, defines sample rate of audio time series .

			If audio is None and audioPath is not None, defines sample rate to load the audio at.

		audioPath	: str, int, pathlib.Path, file-like object or None
			Path to the input file.

			Used only if audio is None. Audio is loaded as mono.

			Sent to ``librosa.load()`` as ``path`` parameter.

			If None, ``audio`` cannot be None.

		startTime	: float
			Time stamp to consider audio from.

		duration	: float or None; default=None
			If duration is None
				- If ``audio`` is None, duration is inferred from the audio.
				- If ``audio`` is None and ``audioPath`` is not None, the entire song is loaded.
		
		onsetFile	: str
			File path to csv onset time stamps.

		onsetLabels	: str or list
			Column name(s) in onsetFile with time stamp for different types of onsets.

			If a list is given, then clicks will be generated with different frequencies for each column name in the list.

		destPath	: str or None
			Path to save audio file at.
			
			If None, will not save any audio file.

	Returns
	-------
		iPython.display.Audio 
			Object that plays the audio with clicks.
	'''

	if audio is None:
		audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=duration)
	if duration is None:
		duration = librosa.get_duration(audio)
	onsetFileVals = pd.read_csv(onsetFile)
	onsetTimes = []

	# check if onsetLabels is a str, convert it to a list
	if isinstance(onsetLabels, str):
		onsetLabels = [onsetLabels]
	for onsetLabel in onsetLabels:
		onsetTimes.append(onsetFileVals.loc[(onsetFileVals[onsetLabel] >= startTime) & (onsetFileVals[onsetLabel] <= startTime+duration), onsetLabel].values)
	clickTracks = [librosa.clicks(onsetTime-startTime, sr=sr, length=len(audio), click_freq=1000*(2*i+1)) for i, onsetTime in enumerate(onsetTimes)] 	# frequency of each click is set 2000 Hz apart.
	audioWClicks = 0.8*audio  # scale audio amplitude by 0.8
	for clickTrack in clickTracks:
		audioWClicks += 0.2/len(clickTracks)*clickTrack 	# add clicks to the audio
	if destPath is not None:
		# write the audio
		sf.write(destPath, audioWClicks, sr)
	return Audio(audioWClicks, rate=sr)

# AUDIO MANIPULATION
def playVideo(video=None, videoPath=None, startTime=0, duration=None, destPath='Data/Temp/VideoPart.mp4', videoOffset=0):
	'''Plays relevant part of a given video.

	If ``duration`` is None and ``startTime`` is 0, the entire Video is returned. 
	
	If ``duration`` is not None or ``startTime`` is not 0, the video is cut using the ``ffmpeg`` Python library and is stored in ``destPath``. 

	Parameters
	----------
		video	: ndarray or None
			Loaded video sample. 

			When ``video`` is not None, all the other parameters in the function are not considered. If a trimmed video is needed, please use ``videoPath`` instead.

			If None, ``videoPath`` will be used to load the video.

		videoPath	: str
			Path to video file.

			Passed to ``data`` parameter in ``Video()``.

		startTime	: float
			Time to start reading the video from. 
			
			Used only when ``video`` is None.
		
		duration	: float
			Duration of the video to load.

			Used only when ``video`` is None.

		destPath	: str or None
			Path to store shortened video.

			Used only when ``video`` is None.

		videoOffset	: float
			Number of seconds offset between video and audio files. This parameter is useful when the video is present only for an excerpt of the audio file.
			
			::
				time in audio + ``videoOffset`` = time in video
	Returns
	-------
		iPython.display.Video 
			Object that plays the video.

	Raises
	------
		ValueError
			If ``destPath`` is None, when ``startTime`` != 0 or ``duration`` is not None.
	'''
	if video is None:
		if duration is None and startTime == 0:
			# play the entire video
			return Video(videoPath, embed=True)
		else:
			# store a shortened video in destPath
			if destPath is None:
				# if destPath is None, raise an error
				raise ValueError(f'destPath cannot be None if video is to be trimmed before playing. destPath has invalid type of {type(destPath)}.')
			vid = ffmpeg.input(videoPath)
			joined = ffmpeg.concat(
			vid.video.filter('trim', start=startTime+videoOffset, duration=duration).filter('setpts', 'PTS-STARTPTS'),
			vid.audio.filter('atrim', start=startTime+videoOffset, duration=duration).filter('asetpts', 'PTS-STARTPTS'),
			v=1,
			a=1
			).node
			v3 = joined['v']
			a3 = joined['a']
			out = ffmpeg.output(v3, a3, destPath).overwrite_output()
			out.run()
			return Video(destPath, embed=True)
	else:
		return Video(data=video, embed=True)

# PLOTTING FUNCTION
def generateFig(noRows, figSize=(14, 7), heightRatios=None):
	'''Generates a matplotlib.pyplot.figure and axes to plot in.

	Axes in the plot are stacked vertically in one column, with height of each axis determined by heightRatios.

	Parameters
	----------
		noRows	: int > 0
			Number of plots (i.e. rows) in the figure.

		figSize	: (float, float)
			(width, height) in inches of the figure.

			Passed to ``matplotlib.figure.Figure()``.

		heightRatios	: list or None
			List of heights that each plot in the figure should take. Relative height of each row is determined by ``heightRatios[i] / sum(heightRatios)``.

			Passed to ``matplotlib.figure.Figure.add_gridspec()`` as the parameter ``height_ratios``.
			
			.. note ::
				len(heightRatios) has to be equal to noRows

	Returns
	-------
		fig	: matplotlib.figure.Figure()
			Figure object with all the plots

		axs	: list of matplotlib.axes.Axes objects 
			List of axes objects. Each object corresponds to one row/plot. 

	Raises
	------
		Exception
			If ``len(heightRatios) != noRows``.
	'''
	if len(heightRatios) != noRows:
		Exception("Length of heightRatios has to be equal to noRows")

	fig = plt.figure(figsize=figSize)
	specs = fig.add_gridspec(noRows, 1, height_ratios = heightRatios)
	axs = [fig.add_subplot(specs[i, 0]) for i in range(noRows)]
	return fig, axs


def subBandEner(X,fs,band):
	'''Computes spectral sub-band energy by summing squared magnitude values of STFT over specified spectral band (suitable for vocal onset detection).

	Parameters
	----------
		X   : ndarray
			STFT of an audio signal x
		fs  : int or float
			Sampling rate
		band    : list or tuple or ndarray
			Edge frequencies (in Hz) of the sub-band of interest
		
	Returns
	----------
		sbe : ndarray
			Array with each value representing the magnitude STFT values in a short-time frame squared & summed over the sub-band
	'''

	#convert band edge frequencies to bin numbers
	binLow = int(np.ceil(band[0]*X.shape[0]/(fs/2)))
	binHi = int(np.ceil(band[1]*X.shape[0]/(fs/2)))

	#compute sub-band energy
	sbe = np.sum(np.abs(X[binLow:binHi])**2, 0)

	return sbe

def biphasicDerivative(x, hopDur, norm=True, rectify=True):
	'''Computes the biphasic derivative of a signal(See [1]_ for a detailed explanation of the algorithm).

	.. [1] Rao, P., Vinutha, T.P. and Rohit, M.A., 2020. Structural Segmentation of Alap in Dhrupad Vocal Concerts. Transactions of the International Society for Music Information Retrieval, 3(1), pp.137–152. DOI: http://doi.org/10.5334/tismir.64
	
	Parameters
	----------
		x   : ndarray
			Input signal
		hopDur  : float
			Sampling interval in seconds of input signal (reciprocal of sampling rate of x)
		norm    :bool
			If output is to be normalized
		rectify :bool
			If output is to be rectified to keep only positive values (sufficient for peak-picking)
	
	Returns
	----------
		x   : ndarray
			Output of convolving input with the biphasic derivative filter

	'''

	#sampling instants
	n = np.arange(-0.1, 0.1, hopDur)

	#filter parameters (see [1] for explanation)
	tau1 = 0.015  # -ve lobe width
	tau2 = 0.025  # +ve lobe width
	d1 = 0.02165  # -ve lobe position
	d2 = 0.005  # +ve lobe position

	#filter
	A = np.exp(-pow((n-d1)/(np.sqrt(2)*tau1), 2))/(tau1*np.sqrt(2*np.pi))
	B = np.exp(-pow((n+d2)/(np.sqrt(2)*tau2), 2))/(tau2*np.sqrt(2*np.pi))
	biphasic = A-B

	#convolve with input and invert
	x = np.convolve(x, biphasic, mode='same')
	x = -1*x

	#normalise and rectify
	if norm:
		x/=np.max(x)
		x-=np.mean(x)

	if rectify:
		x*=(x>0)

	return x

def toDB(x, C):
	'''Applies logarithmic (base 10) transformation (based on [1])
	
	Parameters
	----------
		x   : ndarray
			Input signal
		C   : int or float
			Scaling constant
	
	Returns
	----------
		log-scaled x
	'''
	return np.log10(1 + x*C)/(np.log10(1+C))

def getODF(audio=None, audioPath=None, startTime=0, duration=None, fs=16000, winSize=640, hopSize=160, nFFT=1024, source='vocal'):
	'''Computes onset activation function from audio signal using short-time spectrum based methods.

	Parameters
	----------
		audio   : ndarray
			Audio signal
		audioPath  : str
			Path to the audio file
		startTime   : int or float
			Time to start reading the audio at
		duration    : int or float
			Duration of audio to read
		fs  : int or float
			Sampling rate to read audio at
		winSize : int
			Window size (in frames) for STFT
		hopSize : int
			Hop size (in frames) for STFT
		nFFT    : int
			DFT size
		source  : str
			Source instrument in audio - one of 'vocal' or 'perc' (percussion)

	Returns
	----------
		odf : ndarray
			Frame-wise onset activation function (at a sampling rate of 1/hopSize)
		onsets  ndarray
			Time locations of detected onset peaks in the odf (peaks detected using peak picker from librosa)
	'''

	#if audio signal is provided
	if audio is not None:
		#select segment of interest from audio based on start time and duration
		if duration is None:
			audio = audio[int(np.ceil(startTime*fs)):]
		else:
			audio = audio[int(np.ceil(startTime*fs)):int(np.ceil((startTime+duration)*fs))]

	#if audio path is provided
	elif audioPath is not None:
		audio,_ = librosa.load(audioPath, sr=fs, offset=startTime, duration=duration)

	else:
		print('Provide either the audio signal or path to the stored audio file on disk')
		raise

	#fade in and out ends of audio to prevent spurious onsets due to abrupt start and end
	audio = fadeIn(audio,int(0.5*fs))
	audio = fadeOut(audio,int(0.5*fs))

	#compute magnitude STFT
	X,_ = librosa.magphase(librosa.stft(audio, win_length=winSize, hop_length=hopSize, n_fft=nFFT))

	#use sub-band energy -> log transformation -> biphasic filtering, if vocal onset detection [1]
	if source=='vocal':
		sub_band = [600,2400] #Hz
		odf = subBandEner(X, fs, sub_band)
		odf = toDB(odf, 100)
		odf = biphasicDerivative(odf, hopSize/fs, norm=True, rectify=True)

		#get onset locations using librosa's peak-picking function
		onsets = librosa.onset.onset_detect(onset_envelope=odf.copy(), sr=fs, hop_length=hopSize, pre_max=4, post_max=4, pre_avg=6, post_avg=6, wait=50, delta=0.12)*hopSize/fs

	#use spectral flux method (from FMP notebooks [2])
	elif source=='perc':
		sub_band = [0,fs/2] #full band
		odf = fmp.spectral_flux(audio, Fs=fs, N=nFFT, W=winSize, H=hopSize, M=20, band=sub_band)
		onsets = librosa.onset.onset_detect(onset_envelope=odf, sr=fs, hop_length=hopSize, pre_max=1, post_max=1, pre_avg=1, post_avg=1, wait=10, delta=0.05)*hopSize/fs

	return odf, onsets
	
def fadeIn(x, length):
	'''
	Apply fade-in to the beginning of an audio signal using a hanning window.
	
	Parameters
	----------
		x   : ndarray
			Signal
		length  : int
			Length of fade (in samples)
	
	Returns
	----------
		x   : ndarray
			Faded-in signal
	'''
	x[:length] *= (np.hanning(2*length))[:length]
	return x

def fadeOut(x, length):
	'''
	Apply fade-out to the end of an audio signal using a hanning window.
	
	Parameters
	----------
		x   : ndarray
			Signal
		length  : int
			Length of fade (in samples)
	
	Returns
	----------
		x   : ndarray
			Faded-out signal
	'''
	x[-length:] *= np.hanning(2*length)[length:]
	return x

def autoCorrelationFunction(x, fs, maxLag, winSize, hopSize):
	'''
	Compute short-time autocorrelation of a signal and normalise every frame by the maximum correlation value.
	
	Parameters
	----------
		x   : ndarray
			Input signal
		fs  : int or float
			Sampling rate
		maxLag  : int or float
			Maximum lag in seconds upto which correlation is to be found (ACF is computed for all unit sample shift values lesser than this limit)
		winSize : int or float
			Length in seconds of the signal selected for short-time autocorrelation
		hopSize : int or float
			Hop duration in seconds between successive windowed signal segments (not the same as lag/shift)
	
	Returns
	----------
		ACF : ndarray
			short-time ACF matrix [shape=(#frames,#lags)]
	'''

	#convert parameters to frames from seconds
	n_ACF_lag = int(maxLag*fs)
	n_ACF_frame = int(winSize*fs)
	n_ACF_hop = int(hopSize*fs)

	#split input signal into windowed segments
	x = subsequences(x, n_ACF_frame, n_ACF_hop)

	#compute ACF for each windowed segment
	ACF = np.zeros((len(x), n_ACF_lag))
	for i in range(len(ACF)):
		ACF[i][0] = np.dot(x[i], x[i])
		for j in range(1, n_ACF_lag):
			ACF[i][j] = np.dot(x[i][:-j], x[i][j:])

	#normalise each ACF vector (every row) by max value
	for i in range(len(ACF)):
		if max(ACF[i])!=0:
			ACF[i] = ACF[i]/max(ACF[i])

	return ACF

def subsequences(x, winSize, hopSize):
	'''
	Split signal into shorter windowed segments with a specified hop between consecutive windows.
	
	Parameters
	----------
		x   : ndarray
			Input signal
		winSize : int or float
			Size of short-time window in seconds
		hopSize : int or float
			Hop duration in seconds between consecutive windows
	
	Returns
	----------
		x_sub   : ndarray
			2d array containing windowed segments
	'''

	#pre-calculate shape of output numpy array (#rows based on #windows obtained using provided window and hop sizes)
	shape = (int(1 + (len(x) - winSize)/hopSize), winSize)

	strides = (hopSize*x.strides[0], x.strides[0])
	x_sub = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
	return x_sub

def tempoPeriodLikelihood(ACF, norm=True):
	'''
	Compute from ACF, the likelihood of each ACF lag being the time-period of the tempo. Likelihood is obtained by taking a dot product between the ACF vector and a comb-filter (see [3] for details) in each time frame.
	
	Parameters
	----------
		ACF : ndarray
			Short-time ACF matrix of a signal
		norm    : bool

	Returns
	----------
		tempo_period_candidates : ndarray
			2d array with a likelihood value for each lag in each time frame
	'''
	L = np.shape(ACF)[1]
	N_peaks = 11 # ?

	window = np.zeros((L, L))
	for j in range(L):
		C = j*np.arange(1, N_peaks)
		D = np.concatenate((C, C+1, C-1, C+2, C-2, C+3, C-3))
		D = D[D<L]
		norm_factor = len(D)
		if norm:
			window[j][D] = 1.0/norm_factor
		else:
			window[j][D] = 1.0
			
	tempo_period_candidates = np.dot(ACF, np.transpose(window))
	return tempo_period_candidates

def viterbiSmoothing(tempoPeriodLikelihood, hopDur, transitionPenalty, tempoRange=(30,100)):
	'''
	Apply viterbi smoothing on tempo period (lag) likelihood values to find optimum sequence of tempo values across audio frames (based on [3]).
	
	Parameters
	----------
		tempoPeriodLikelihood   : ndarray
			Likelihood values at each lag (tempo period)
		hopDur  : int or float
			Short-time analysis hop duration in seconds between samples of ACF vector (not hop duration between windowed signal segments taken for ACF)
		transitionPenalty   : int or float
			Penalty factor multiplied with the magnitude of tempo change between frames; high value penalises larger jumps more, suitable for metric tempo that changes gradually across a concert and not abruptly
		tempoRange  : tuple or list
			Expected min and max tempo in BPM

	Returns
	----------
		tempo_period_smoothed   : ndarray
			Array of chosen tempo period in each time frame
	'''

	#convert tempo range to tempo period (in frames) range
	fs = 1/hopDur
	tempoRange = np.around(np.array(tempoRange)*(fs/60)).astype(int)

	#initialise cost matrix with very high values
	T,L = np.shape(tempoPeriodLikelihood)
	cost = np.ones((T,L))*1e8

	#matrix to store cost-minimizing lag in previous frame to each lag in current time frame
	m = np.zeros((T,L))

	#compute cost value at each lag (within range), at each time frame
	#loop over time frames
	for i in range(1,T):
		#loop over lags in current time frame
		for j in range(*tempoRange):
			#loop over lags in prev time frame
			for k in range(*tempoRange):
				if cost[i][j]>cost[i-1][k]+transitionPenalty*abs(60.0*fs/j-60.0*fs/k)-tempoPeriodLikelihood[i][j]:
					#choose lag 'k' in prev time frame that minimizes cost at lag 'j' in current time frame 
					cost[i][j]=cost[i-1][k]+transitionPenalty*abs(60.0*fs/j-60.0*fs/k)-tempoPeriodLikelihood[i][j]
					m[i][j]=int(k)

	#determine least cost path - start at the last frame (pick lag with minimum cost)
	tempo_period_smoothed = np.zeros(T)
	tempo_period_smoothed[T-1] = np.argmin(cost[T-1,:])/float(fs)
	t = int(m[T-1,np.argmin(cost[T-1,:])])

	#loop in reverse till the first frame, reading off values in 'm[i][j]'
	i = T-2
	while(i>=0):
		tempo_period_smoothed[i]=t/float(fs)
		t=int(m[i][t])
		i=i-1
	return tempo_period_smoothed
	
# COMPUTATION FUNCTION
def intensityContour(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, minPitch=98, timeStep=0.01, ax=None, **kwargs):
	'''Calculates the intensity contour for an audio clip.

	Intensity contour is generated for a given audio with [1]_ and it's Python API [2]_. 

	.. [1] Boersma, P., & Weenink, D. (2021). Praat: doing phonetics by computer [Computer program]. Version 6.1.38, retrieved 2 January 2021 from http://www.praat.org/

	.. [2] Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

	Uses ``plotIntensity()`` to plot the contour is ``ax`` is not None. 

	Parameters
	----------
		audio	: ndarray or None
			Loaded audio time series

		sr	: number > 0; default=16000
			If audio is not None, defines sample rate of audio time series 

			If audio is None and audioPath is not None, defines sample rate to load the audio at

		audioPath	: str, int, pathlib.Path, file-like object or None
        	Path to the input file.

			Used only if audio is None. Audio is loaded as mono.

			Sent to ``librosa.load()`` as ``path`` parameter.

			If None, ``audio`` cannot be None.

		startTime    : float; default=0
			Time stamp to consider audio from

		duration    : float or None; default=None
			If duration is None
				- If ``audio`` is None, duration is inferred from the audio.
				- If ``audio`` is None and ``audioPath`` is not None, the entire song is loaded.

		minPitch    : float; default=98
			Minimum pitch (in Hz) to read for contour extraction.

			Passed as ``minimum_pitch`` parameter to ``parselmouth.Sound.to_intensity()``.

		timeStep    : float; default=0.01
			Time steps (in seconds) in which pitch values are extracted. ::

				Example: timeStep = 0.01 implies that pitch values are extracted for every 0.01 s.

			Passed as ``time_step`` parameter to ``parselmouth.Sound.to_intensity()``.

		ax    : matplotlib.axes.Axes or None
			Axes object to plot the intensity contour in.

			If None, will return a tuple with (intensity contour, time steps)

		kwargs	: Additional arguements passed to ``plotIntensity()``.

	Returns
	-------
		ax : matplotlib.axes.Axes
			Plot of intensity contour if ``ax`` was not None

		(intensityVals, t)    : (ndarray, ndarray)
			Tuple with arrays of intensity values (in dB) and time stamps. Returned if ax was None.
	
	'''
	startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
	duration = math.ceil(duration)  # set duration to an integer, for better readability on the x axis of the plot
	if audio is None:
		# if audio is not given, load audio from audioPath
		audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration = duration)
	snd = parselmouth.Sound(audio, sr)
	intensity = snd.to_intensity(time_step=timeStep, minimum_pitch=minPitch)
	intensityVals = intensity.values[0]
	t = intensity.xs() + startTime
	
	if ax is None:
		# if ax is None, return intensity and time values
		return (intensityVals, t)
	else:
		# else plot the contour
		return plotIntensity(intensityVals=intensityVals, t=t, ax=ax, startTime=startTime, duration=duration, **kwargs)

# PLOTTING FUNCTION
def plotIntensity(intensityVals=None, t=None, ax=None, startTime=0, duration=None, freqXlabels=5, xticks=False, yticks=True, xlabel=False, ylabel=True, title='Intensity Contour', annotate=False, ylim=None, c='yellow', **kwargs):
	'''Function to plot a computed intensity contour from ``intensityContour()`` function. 

	Parameters
	----------
		intensityVals    : ndarray
			Intensity contour from ``intensityContour()``

		t    : ndarray
			Time steps corresponding to the ``intensityVals`` from ``intensityContour``

		ax    : matplotlib.axes.Axes or None
			Object on which intensity contour is to be plotted

			If None, will plot in ``matplotlib.pyplot.gca()``.

		startTime    : float >= 0
			Offset time (in seconds) from where audio is analysed.

			Sent to ``drawAnnotation()``.

		duration    : float >= 0 or None
			Duration of audio in the plot.

			If None, will consider the entire audio.

			Sent to ``drawAnnotation()``.

		freqXlabels    : int
			Time (in seconds) after which each x ticklabel should occur

		annotate    : bool
			If true will mark annotations provided in the plot.

			Send to ``drawAnnotation()``.

		ylim    : (float, float) or None
			(min, max) limits for the y axis.
			
			If None, will be directly interpreted from the data.

		c	: color
			Specifies the colour of the plotted intensity contour.

		kwargs	: Additional arguements passed to ``drawAnnotation()`` if ``annotate`` is True.

	Returns
	-------
		ax    : matplotlib.axes.Axes
			Plot of intensity contour.

	Raises
	------
		ValueError
			If intensityVals is not given
	
	'''
	if intensityVals is None:
		Exception('No intensity contour provided')
	
	# check if ax is None
	ax = __check_axes(ax)
	
	ax = sns.lineplot(x=t, y=intensityVals, ax=ax, color=c);
	ax.set(xlabel='Time (s)' if xlabel else '', 
	ylabel='Intensity (dB)' if ylabel else '', 
	title=title, 
	xlim=(startTime, duration+startTime), 
	xticks=np.around(np.arange(math.ceil(startTime), math.floor(startTime+duration), freqXlabels)).astype(int),     # start the xticks such that each one corresponds to an integer with xticklabels
	xticklabels=np.around(np.arange(math.ceil(startTime), math.floor(startTime+duration), freqXlabels)).astype(int) if xticks else [], 	# let the labels start from the integer values.
	ylim=ylim if ylim is not None else ax.get_ylim())
	if not yticks:
		ax.set(yticklabels=[])
	if annotate:
		ax = drawAnnotation(startTime=startTime, duration=duration, ax=ax, **kwargs)
	return ax

# PLOTTING FUNCTION
def plot_hand(annotationFile=None, startTime=0, duration=None, vidFps=25, ax=None, freqXlabels=5, xticks=False, yticks=False, xlabel=False, ylabel=True, title='Wrist Position Vs. Time', vidOffset=0, lWristCol='LWrist', rWristCol='RWrist', wristAxis='y', annotate=False, ylim=None, **kwargs):
	'''Function to plot hand movement.

	Using Openpose annotations, this function plots the height of each hand's wrist vs time. 

	If ``ax`` is None, this will on ``plt.gca()``, i.e. the current axes being used
	
	Parameters
	----------
		annotationFile    : str
			File path to Openpose annotations.

		startTime    : float
			Start time for x labels in the plot (time stamp with respect to the audio signal).

		duration    : float
			Duration of audio to consider for the plot.
			
		vidFps    : float
			FPS of the video data used in openpose annotation.

		ax    : matplotlib.axes.Axes or None 
		Axes object on which plot is to be plotted.

		If None, uses the current Axes object in use with ``plt.gca()``. 

		freqXlabels    : int > 0 
			Time (in seconds) after which each x ticklabel occurs

		xticks    : bool
			If True, will add xticklabels to plot.

		yticks    : bool
			If True, will add yticklabels to plot.

		xlabel    : bool
			If True, will print xlabel in the plot.

		ylabel    : bool
			If True will print ylabel in the plot.

		title    : str
			Title to add to the plot.

		videoOffset    : float
			Number of seconds offset between video and audio ::
				time in audio + videioOffset = time in video

		lWristCol    : str
			Name of the column with left wrist data in ``annotationFile``.

		rWristCol    : str
			Name of the column with right wrist data in ``annotationFile``.

		wristAxis    : str
			Level 2 header in the ``annotationFile`` denoting axis along which movement is plotted (x, y or z axes).

		annotate    : bool
			If True will mark annotations provided on the plot.

		ylim    : (float, float) or None
			(min, max) limits for the y axis.
			
			If None, will be directly interpreted from the data.

		kwargs	: Additional arguements passed to ``drawAnnotation()`` if ``annotate`` is True.
		
	Returns
	-------
		ax    : matplotlib.axes.Axes
			Axes object with plot

	'''
	startTime = startTime + vidOffset   # convert startTime from time in audio to time in video. See parameter definition of ``videoOffset`` for more clarity.
	duration = duration
	movements = pd.read_csv(annotationFile, header=[0, 1])
	lWrist = movements[lWristCol][wristAxis].values[startTime*vidFps:int((startTime+duration)*vidFps)]
	rWrist = movements[rWristCol][wristAxis].values[startTime*vidFps:int((startTime+duration)*vidFps)]
	xvals = np.linspace(startTime, startTime+duration, vidFps*duration, endpoint=False)

	# if ax is None, use plt.gca()
	ax = __check_axes(ax)
	ax.plot(xvals, lWrist, label='Left Wrist')
	ax.plot(xvals, rWrist, label='Right Wrist')
	ax.set(xlabel='Time (s)' if xlabel else '', 
	ylabel='Wrist Position' if ylabel else '', 
	title=title, 
	xlim=(startTime, startTime+duration), 
	xticks=np.around(np.arange(math.ceil(startTime), math.floor(startTime+duration), freqXlabels)).astype(int),     # start the xticks such that each one corresponds to an integer with xticklabels
	xticklabels=np.around(np.arange(math.ceil(startTime), math.floor(startTime+duration), freqXlabels)).astype(int) if xticks else [], 	# let the labels start from the integer values
	ylim=ylim if ylim is not None else ax.get_ylim()
	)
	if not yticks:
		ax.set(yticklabels=[])
	ax.invert_yaxis()    # inverst y-axis to simulate the height of the wrist that we see in real time
	ax.legend()
	if annotate:
		ax = drawAnnotation(startTime=startTime-vidOffset, duration=duration, ax=ax, **kwargs)
	return ax

# ANNOTATION FUNCTION
def annotateInteraction(axs, keywords, cs, interactionFile, startTime, duration):
	'''Adds interaction annotation to the axes given. 

	Used in fig 3.

	Parameters
	----------
		axs    : list of matplotlib.axes.Axes objects
			List of objects to add annotation to.

		keywords    : list
			Keyword corresponding to each Axes object. Value appearing in the 'Type' column in ``interactionFile``. 
			
			.. note ::
				If len(keywords) = len(axs) + 1, the last keyword is plotted in all Axes objects passed.

		cs    : list 
			List of colours associated with each keyword.

		interactionFile    : str
			Path to csv file with the annotation of the interactions.

		startTime    : float >= 0
			Time to start reading the audio at.

		duration    : float >= 0 
			Length of audio to consider.

	Returns
	-------
		axs    : matplotlib.axes.Axes
			List of axes with annotation of interaction
	'''

	annotations = pd.read_csv(interactionFile, header=None)
	annotations.columns = ['Type', 'Start Time', 'End Time', 'Duration', 'Label']
	annotations = annotations.loc[((annotations['Start Time'] >= startTime) & (annotations['Start Time'] <= startTime+duration)) &
								((annotations['End Time'] >= startTime) & (annotations['End Time'] <= startTime+duration))
								]
	for i, keyword in enumerate(keywords):
		lims = axs[i].get_ylim()
		if i < len(axs):
			# keyword corresponds to a particular axis
			for j, annotation in annotations.loc[annotations['Type'] == keyword].iterrows():
				lims = axs[i].get_ylim()
				axs[i].annotate('', xy=(annotation['Start Time'], 0.25*(lims[1] - lims[0]) + lims[0]), xytext=(annotation['End Time'], 0.25*(lims[1] - lims[0]) + lims[0]), arrowprops={'headlength': 0.4, 'headwidth': 0.2, 'width': 3, 'ec': cs[i], 'fc': cs[i]})
				axs[i].annotate(annotation['Label'], (annotation['Start Time'] +annotation['Duration']/2, 0.3*(lims[1] - lims[0]) + lims[0]), ha='center')
		else:
			# keyword corresponds to all axes
			for ax in axs:
				for _, annotation in annotations.loc[annotations['Type'] == keyword].iterrows():
					for j, annotation in annotations.loc[annotations['Type'] == keyword].iterrows():
						ax.annotate('', xy=(annotation['Start Time'], 0.75*(lims[1] - lims[0]) + lims[0]), xytext=(annotation['End Time'], 0.75*(lims[1] - lims[0]) + lims[0]), arrowprops={'headlength': 0.4, 'headwidth': 0.2, 'width': 3, 'ec':cs[i], 'fc': cs[i]})
						ax.annotate(annotation['Label'], (annotation['Start Time'] + annotation['Duration']/2, 0.8*(lims[1] - lims[0]) + lims[0]), ha='center')  
	return axs

def drawHandTap(ax, handTaps, c='purple'):
	'''Plots the hand taps as vertical lines on the Axes object ``ax``. 
	
	Used in fig 9.
	
	Parameters
	----------
		ax    : matplotlib.axes.Axes or None
			Axes object to add hand taps to

			If None, will plot on ``plt.gca()``.

		handTaps    : ndarray
			Array of hand tap timestamps.

		c    : color
			Color of the line

			Passed to ``plt.axes.Axes.axvline()``.

	Returns
	-------
		matplotlib.axes.Axes
			Plot with lines
	'''
	for handTap in handTaps:
		ax.axvline(handTap, linestyle='--', c=c, alpha=0.6)
	return ax

# AUDIO MANIPULATION
def generateVideoWSquares(vid_path, tapInfo, dest_path='Data/Temp/vidWSquares.mp4', vid_size=(720, 576)):
	'''Function to genrate a video with rectangles for each hand tap. 
	
	Used in fig 9.
	
	Parameters
	----------
		vid_path    : str
			
			Path to the original video file.

		tapInfo    : list
			
			List of metadata associated with each handtap.
			
			Metadata for each handtap consists of: 
				- time    : float
					time stamp of hand tap (in seconds).
				- keyword    : str    
					keyword specifying which hand tap to consider
				- (pos1, pos2)    : ((float, float), (float, float))
					(x, y) coordinates of opposite corners of the box to be drawn.
				- color    : (int, int, int) or color
					If (int, int, int) then it is a tuple with RGB values associated with the colour.

		dest_path    : str
			
			File path to save video with squares.

		vid_size    : (int, int)
			
			(width, height) of video to generate in pixels
	Returns
		None
	'''

	cap_vid = cv2.VideoCapture(vid_path)
	fps = cap_vid.get(cv2.CAP_PROP_FPS)
	framesToDraw = defaultdict(list)   # dictionary with frame numbers as keys and properties of square box to draw as list of values
	for ind, timeRow in enumerate(tapInfo):
		if is_color_like(timeRow[3]):
			# pdb.set_trace()
			tapInfo[ind][3] = [int(x*255) for x in list(to_rgb(timeRow[3]))]
		# converts data from RGB to BGR
		tapInfo[ind][3] = tuple([int(x) for x in timeRow[3]][::-1])
		framesToDraw[int(np.around(timeRow[0]*fps))] = timeRow[1:]
	output = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(*"XVID"), fps, vid_size)
	i = 0
	# generate video
	while(cap_vid.isOpened()):
		ret, frame = cap_vid.read()
		if ret == True:
			i+=1
			if i in framesToDraw.keys():
				frame = cv2.rectangle(frame, framesToDraw[i][1][0], framesToDraw[i][1][1], framesToDraw[i][2], 3)
			output.write(frame)
		else:
			# all frames are read
			break
	cap_vid.release()
	output.release()

def combineAudioVideo(vid_path='Data/Temp/vidWSquares.mp4', audio_path='audioWClicks.wav', dest_path='Data/Temp/FinalVid.mp4'):
	'''Function to combine audio and video into a single file. 
	
	Used in fig 9.

	Parameters
	----------
		vid_path    : str
			File path to the video file with squares.
		
		audio_path    : str
			File path to the audio file with clicks.

		dest_path    : str
			File path to store the combined file at.

	Returns
	-------
		None

	'''
	
	vid_file = ffmpeg.input(vid_path)
	audio_file = ffmpeg.input(audio_path)
	(
		ffmpeg
		.concat(vid_file.video, audio_file.audio, v=1, a=1)
		.output(dest_path)
		.overwrite_output()
		.run()
	)
	print('Video saved at ' + dest_path)

def generateVideo(annotationFile, onsetKeywords, vidPath='Data/Temp/VS_Shree_1235_1321.mp4', tempFolder='Data/Temp/', pos=None, cs=None):
	'''Function to generate video with squares and clicks corresponding to hand taps. 
	
	Used in fig 9.
	
	Parameters
	----------
		annotationFile    : str
			File path to the annotation file with hand tap timestamps

		onsetKeywords    : list
			List of column names to read from ``annotationFile``.

		vidPath    : str
			File path to original video file.

		tempFolder    : str
			File path to temporary directory to store intermediate audio and video files in.

		pos    : list
			list of [pos1, pos2] -> 2 opposite corners of the box for each keyword 

		cs    : list
			list of [R, G, B] colours used for each keyword
	
	Returns
		None
	'''
	annotations = pd.read_csv(annotationFile)
	timeStamps = []
	for i, keyword in enumerate(onsetKeywords):
		for timeVal in annotations[keyword].values[~np.isnan(annotations[keyword].values)]:
			timeStamps.append([timeVal, keyword, pos[i], cs[i]])
	timeStamps.sort(key=lambda x: x[0])

	# generate video 
	generateVideoWSquares(vid_path=vidPath, tapInfo=timeStamps, dest_path=os.path.join(tempFolder, 'vidWSquares.mp4'))

	# generate audio
	playAudioWClicks(audioPath=vidPath, onsetFile=annotationFile, onsetLabels=onsetKeywords, destPath=os.path.join(tempFolder, 'audioWClicks.wav'))

	# combine audio and video
	combineAudioVideo(vid_path=os.path.join(tempFolder, 'vidWSquares.mp4'), audio_path=os.path.join(tempFolder, 'audioWClicks.wav'), dest_path=os.path.join(tempFolder, 'finalVid.mp4'))

'''
References
[1] Rao, P., Vinutha, T.P. and Rohit, M.A., 2020. Structural Segmentation of Alap in Dhrupad Vocal Concerts. Transactions of the International Society for Music Information Retrieval, 3(1), pp.137–152. DOI: http://doi.org/10.5334/tismir.64
[2] Meinard Müller and Frank Zalkow: FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing. Proceedings of the International Conference on Music Information Retrieval (ISMIR), Delft, The Netherlands, 2019.
[3] T.P. Vinutha, S. Suryanarayana, K. K. Ganguli and P. Rao " Structural segmentation and visualization of Sitar and Sarod concert audio ", Proc. of the 17th International Society for Music Information Retrieval Conference (ISMIR), Aug 2016, New York, USA
'''
