import numpy as np
import librosa
import scipy.signal as signal
from matplotlib import pyplot as plt

def annotationsToFMPFormat(annotPath, duration):
    """
    Converts 2 column (Time, Label) format into FMP format ([start time, end time, label])
    
    Parameters:
        annot: path to annotations file
        duration: duration of audio to use in last row of converted annotation
    
    Returns:
        annotFMP: annotations in FMP format        
    """
    annot = pd.read_csv(annotPath)
    annotFMP = []
    for i in range(annot.shape[0]-1):
        annotFMP.append([annot.iloc[i,0], annot.iloc[i+1,0], annot.iloc[i,-1]])

    endTime = annot.iloc[0,0]+duration
    annotFMP.append([annot.iloc[-1,0], endTime, annot.iloc[-1,-1]])

    return annotFMP

def smooth_downsample_feature_sequence(X, Fs, filtLen=41, downSampling=10, window='boxcar'):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by convolution with a filter kernel

    Notebook: C3/C3S1_FeatureSmoothing.ipynb

    Args:
        X: Feature sequence
        Fs: Frame rate of `X`
        filtLen: Length of smoothing filter
        downSampling: Downsampling factor
        window: Window type of smoothing filter

    Returns:
        X_smooth: Smoothed and downsampled feature sequence
        Fs_feature: Frame rate of `X_smooth`
    """
    filt_kernel = np.expand_dims(signal.get_window(window, filtLen), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode='same') / filtLen
    X_smooth = X_smooth[:, ::downSampling]
    Fs_feature = Fs / downSampling
    return X_smooth, Fs_feature

def threshold_matrix(S, thresh, strategy='absolute', scale=False, penalty=0, binarize=False):
    """Treshold matrix in a relative fashion

    Notebook: C4/C4/C4S2_SSM-Thresholding.ipynb

    Args:
        S: Input matrix
        thresh: Treshold (meaning depends on strategy)
        strategy: Thresholding strategy ('absolute', 'relative', 'local')
        scale: If scale=True, then scaling of positive values to range [0,1]
        penalty: Set values below treshold to value specified
        binarize: Binarizes final matrix (positive: 1; otherwise: 0)
        Note: Binarization is applied last (overriding other settings)


    Returns:
        S_thresh: Thresholded matrix
    """
    if np.min(S)<0:
        raise Exception('All entries of the input matrix must be nonnegative')

    S_thresh = np.copy(S)
    N, M = S.shape
    num_cells = N*M

    if strategy == 'absolute':
        thresh_abs = thresh
        S_thresh[S_thresh < thresh] = 0

    if strategy == 'relative':
        thresh_rel = thresh
        num_cells_below_thresh = int(np.round(S_thresh.size*(1-thresh_rel)))
        if num_cells_below_thresh < num_cells:
            values_sorted = np.sort(S_thresh.flatten('F'))
            thresh_abs = values_sorted[num_cells_below_thresh]
            S_thresh[S_thresh < thresh_abs] = 0
        else:
            S_thresh = np.zeros([N,M])

    if strategy == 'local':
        thresh_rel_row = thresh[0]
        thresh_rel_col = thresh[1]
        S_binary_row = np.zeros([N,M])
        num_cells_row_below_thresh = int(np.round(M*(1-thresh_rel_row)))
        for n in range(N):
            row = S[n,:]
            values_sorted = np.sort(row)
            if num_cells_row_below_thresh < M:
                thresh_abs = values_sorted[num_cells_row_below_thresh]
                S_binary_row[n,:] = (row>=thresh_abs)
        S_binary_col = np.zeros([N,M])
        num_cells_col_below_thresh = int(np.round(N*(1-thresh_rel_col)))
        for m in range(M):
            col = S[:,m]
            values_sorted = np.sort(col)
            if num_cells_col_below_thresh < N:
                thresh_abs = values_sorted[num_cells_col_below_thresh]
                S_binary_col[:,m] = (col>=thresh_abs)
        S_thresh =  S * S_binary_row * S_binary_col

    if scale:
        cell_val_zero = np.where(S_thresh==0)
        cell_val_pos = np.where(S_thresh>0)
        if len(cell_val_pos[0])==0:
            min_value = 0
        else:
            min_value = np.min(S_thresh[cell_val_pos])
        max_value = np.max(S_thresh)
        #print('min_value = ', min_value, ', max_value = ', max_value)
        if max_value > min_value:
            S_thresh = np.divide((S_thresh - min_value) , (max_value -  min_value))
            if len(cell_val_zero[0])>0:
                S_thresh[cell_val_zero] = penalty
        else:
            print('Condition max_value > min_value is voliated: output zero matrix')

    if binarize:
        S_thresh[S_thresh > 0] = 1
        S_thresh[S_thresh < 0] = 0
    return S_thresh
    
def compute_SSM(X, L_smooth=16, tempo_rel_set=np.array([1]), shift_set=np.array([0]), strategy = 'relative', scale=1, thresh= 0.15, penalty=0, binarize=0, feature='chroma'):
    """Compute an SSM

    Notebook: C4S2_SSM-Thresholding.ipynb

    Args:
        X: Feature sequence
        L_smooth, tempo_rel_set, shift_set: Parameters for computing SSM
        strategy, scale, thresh, penalty, binarize: Parameters used for tresholding SSM

    Returns:
        S_thresh, I: SSM and index matrix
    """

    S, I = compute_SM_TI(X, X, L=L_smooth, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)
    S_thresh = threshold_matrix(S, thresh=thresh, strategy=strategy, scale=scale, penalty=penalty, binarize=binarize)
    return S_thresh, I

def compressed_gray_cmap(alpha=5, N=256):
    """Creates a logarithmically or exponentially compressed grayscale colormap

    Notebook: B/B_PythonVisualization.ipynb

    Args:
        alpha: The compression factor. If alpha > 0, it performs log compression (enhancing black colors).
            If alpha < 0, it performs exp compression (enhancing white colors).
            Raises an error if alpha = 0.
        N: The number of rgb quantization levels (usually 256 in matplotlib)

    Returns:
        color_wb: The colormap
    """
    assert alpha != 0

    gray_values = np.log(1 + abs(alpha) * np.linspace(0, 1, N))
    gray_values /= gray_values.max()

    if alpha > 0:
        gray_values = 1 - gray_values
    else:
        gray_values = gray_values[::-1]

    gray_values_rgb = np.repeat(gray_values.reshape(256, 1), 3, axis=1)
    color_wb = LinearSegmentedColormap.from_list('color_wb', gray_values_rgb, N=N)
    return color_wb

def plot_SSM(X, Fs_X, S, Fs_S, ann, duration, color_ann=None,
                               title='', label='Time (seconds)', time=True,
                               figsize=(6, 6), fontsize=10, clim_X=None, clim=None):
    """Plot SSM along with annotations (standard setting is time in seconds)
    Notebook: C4/C4S2_SSM.ipynb
    """
    cmap = compressed_gray_cmap(alpha=-10)
    fig, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                           'wspace': 0.2,
                           'height_ratios': [1, 0.1]},
                           figsize=figsize)
    #plot_matrix(X, Fs=Fs_X, ax=[ax[0,1], ax[0,2]], clim=clim_X,
    #                     xlabel='', ylabel='', title=title)
    #ax[0,0].axis('off')
    plot_matrix(S, Fs=Fs_S, ax=[ax[0,1], ax[0,2]], cmap=cmap, clim=clim,
                         title='', xlabel='', ylabel='', colorbar=True);
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[0,1].grid(False)
    plot_segments(ann, ax=ax[1,1], time_axis=True, fontsize=fontsize,
                           colors=color_ann,
                           time_label=label, time_max=duration)
    ax[1,2].axis('off'), ax[1,0].axis('off')
    plot_segments(ann, ax=ax[0,0], time_axis=True, fontsize=fontsize,
                           direction='vertical', colors=color_ann,
                           time_label=label, time_max=duration)
    return fig, ax

def check_segment_annotations(annot, default_label=''):
    """Checks segment annotation. If label is missing, adds an default label.

    Args:
        annot: A List of the form of [(start_position, end_position, label), ...], or
            [(start_position, end_position), ...]
        default_label: The default label used if label is missing

    Returns:
        annot: A List of tuples in the form of [(start_position, end_position, label), ...]
    """
    assert isinstance(annot[0], (list, np.ndarray, tuple))
    len_annot = len(annot[0])
    assert all(len(a) == len_annot for a in annot)
    if len_annot == 2:
        annot = [(a[0], a[1], default_label) for a in annot]

    return annot

def color_argument_to_dict(colors, labels_set, default='gray'):
    """Creates a color dictionary

    Args:
        colors: Several options: 1. string of FMP_COLORMAPS, 2. string of matplotlib colormap, 3. list or np.ndarray of
            matplotlib color specifications, 4. dict that assigns labels  to colors
        labels_set: List of all labels
        default: Default color, used for labels that are in labels_set, but not in colors

    Returns:
        color_dict: Dictionary that maps labels to colors
    """

    if isinstance(colors, str):
        # FMP colormap
        if colors in FMP_COLORMAPS:
            color_dict = {l: c for l, c in zip(labels_set, FMP_COLORMAPS[colors])}
        # matplotlib colormap
        else:
            cm = plt.get_cmap(colors)
            num_labels = len(labels_set)
            colors = [cm(i / (num_labels + 1)) for i in range(num_labels)]
            color_dict = {l: c for l, c in zip(labels_set, colors)}

    # list/np.ndarray of colors
    elif isinstance(colors, (list, np.ndarray, tuple)):
        color_dict = {l: c for l, c in zip(labels_set, colors)}

    # is already a dict, nothing to do
    elif isinstance(colors, dict):
        color_dict = colors

    else:
        raise ValueError('`colors` must be str, list, np.ndarray, or dict')

    for key in labels_set:
        if key not in color_dict:
            color_dict[key] = default

    return color_dict
    
def plot_segments(annot, ax=None, figsize=(10, 2), direction='horizontal', colors='FMP_1', time_min=None,
                  time_max=None, nontime_min=0, nontime_max=1, time_axis=True, nontime_axis=False, time_label=None,
                  swap_time_ticks=False, edgecolor='k', axis_off=False, dpi=72, adjust_nontime_axislim=True, alpha=None,
                  print_labels=True, label_ticks=False, **kwargs):
    """Creates a multi-line plot for annotation data

    Args:
        annot: A List of tuples in the form of [(start_position, end_position, label), ...]
        ax: The Axes instance to plot on. If None, will create a figure and axes.
        figsize: Width, height in inches
        direction: 'vertical' or 'horizontal'
        colors: Several options: 1. string of FMP_COLORMAPS, 2. string of matplotlib colormap, 3. list or np.ndarray of
            matplotlib color specifications, 4. dict that assigns labels  to colors
        time_min: Minimal limit for time axis. If None, will be min annotation.
        time_max: Maximal limit for time axis. If None, will be max from annotation.
        nontime_min: Minimal limit for non-time axis.
        nontime_max: Maximal limit for non-time axis.
        time_axis: Display time axis ticks or not
        nontime_axis: Display non-time axis ticks or not
        swap_time_ticks: For horizontal: xticks up; for vertical: yticks left
        edgecolor: Color for edgelines of segment box
        axis_off: Calls ax.axis('off')
        dpi: Dots per inch
        adjust_nontime_axislim: Adjust non-time-axis. Usually True for plotting on standalone axes and False for
            overlay plotting
        alpha: Alpha value for rectangle
        print_labels: Print labels inside Rectangles
        label_ticks: Print labels as ticks
        kwargs: Keyword arguments for matplotlib.pyplot.annotate

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
    """
    assert direction in ['vertical', 'horizontal']
    annot = check_segment_annotations(annot)

    if 'color' not in kwargs:
        kwargs['color'] = 'k'
    if 'weight' not in kwargs:
        kwargs['weight'] = 'bold'
        #kwargs['weight'] = 'normal'
    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 12
    if 'ha' not in kwargs:
        kwargs['ha'] = 'center'
    if 'va' not in kwargs:
        kwargs['va'] = 'center'

    if colors is None:
        colors = 'FMP_1'

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    labels_set = sorted(set([label for start, end, label in annot]))
    colors = color_argument_to_dict(colors, labels_set)

    nontime_width = nontime_max - nontime_min
    nontime_middle = nontime_min + nontime_width / 2
    all_time_middles = []

    for start, end, label in annot:
        time_width = end - start
        time_middle = start + time_width / 2
        all_time_middles.append(time_middle)

        if direction == 'horizontal':
            rect = mpatch.Rectangle((start, nontime_min), time_width, nontime_width, facecolor=colors[label],
                                    edgecolor=edgecolor, alpha=alpha)
            ax.add_patch(rect)
            if print_labels:
                ax.annotate(label, (time_middle, nontime_middle), **kwargs)
        else:
            rect = mpatch.Rectangle((nontime_min, start), nontime_width, time_width, facecolor=colors[label],
                                    edgecolor=edgecolor, alpha=alpha)
            ax.add_patch(rect)
            if print_labels:
                ax.annotate(label, (nontime_middle, time_middle), **kwargs)

    if time_min is None:
        time_min = min(start for start, end, label in annot)
    if time_max is None:
        time_max = max(end for start, end, label in annot)

    if direction == 'horizontal':
        ax.set_xlim([time_min, time_max])
        if adjust_nontime_axislim:
            ax.set_ylim([nontime_min, nontime_max])
        if not nontime_axis:
            ax.set_yticks([])
        if not time_axis:
            ax.set_xticks([])
        if swap_time_ticks:
            ax.xaxis.tick_top()
        if time_label:
            ax.set_xlabel(time_label)
        if label_ticks:
            ax.set_xticks(all_time_middles)
            ax.set_xticklabels([label for start, end, label in annot])

    else:
        ax.set_ylim([time_min, time_max])
        if adjust_nontime_axislim:
            ax.set_xlim([nontime_min, nontime_max])
        if not nontime_axis:
            ax.set_xticks([])
        if not time_axis:
            ax.set_yticks([])
        if swap_time_ticks:
            ax.yaxis.tick_right()
        if time_label:
            ax.set_ylabel(time_label)
        if label_ticks:
            ax.set_yticks(all_time_middles)
            ax.set_yticklabels([label for start, end, label in annot])

    if axis_off:
        ax.axis('off')

    if fig is not None:
        plt.tight_layout()

    return fig, ax

def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence

    Notebook: C3/C3S1_FeatureNormalization.ipynb

    Args:
        X: Feature sequence
        norm: The norm to be applied. '1', '2', 'max' or 'z'
        threshold: An threshold below which the vector `v` used instead of normalization
        v: Used instead of normalization below `threshold`. If None, uses unit vector for given norm

    Returns:
        X_norm: Normalized feature sequence
    """
    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'max':
        if v is None:
            v = np.ones(K, dtype=np.float64)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'z':
        if v is None:
            v = np.zeros(K, dtype=np.float64)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v

    return X_norm

def convert_structure_annotation(ann, Fs=1, remove_digits=False, index=False):
    """Convert structure annotations

    Notebook: C4/C4S1_MusicStructureGeneral.ipynb

    Args:
        ann: Structure annotions
        Fs: Sampling rate
        remove_digits: Remove digits from labels

    Returns:
        ann_converted: Converted annotation
    """
    ann_converted = []
    for r in ann:
        s = r[0]*Fs
        t = r[1]*Fs
        if index:
            s = int(np.round(s))
            t = int(np.round(t))
        if remove_digits:
            label = ''.join([i for i in r[2] if not i.isdigit()])
        else:
            label = r[2]
        ann_converted = ann_converted + [[s, t, label]]
    return ann_converted

def compute_SM_dot(X,Y):
    """Computes similarty matrix from feature sequences using dot (inner) product
    Notebook: C4/C4S2_SSM.ipynb
    """
    S = np.dot(np.transpose(Y),X)
    return S

def filter_diag_mult_SM(S, L=1, tempo_rel_set=[1], direction=0):
    """Path smoothing of similarity matrix by filtering in forward or backward direction
    along various directions around main diagonal
    Note: Directions are simulated by resampling one axis using relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S: Self-similarity matrix (SSM)
        L: Length of filter
        tempo_rel_set: Set of relative tempo values
        direction: Direction of smoothing (0: forward; 1: backward)

    Returns:
        S_L_final: Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    num = len(tempo_rel_set)
    S_L_final = np.zeros((M,N))

    for s in range(0, num):
        M_ceil = int(np.ceil(N/tempo_rel_set[s]))
        resample = np.multiply(np.divide(np.arange(1,M_ceil+1),M_ceil),N)
        np.around(resample, 0, resample)
        resample = resample -1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
        S_resample = S[:,index_resample]

        S_L = np.zeros((M,M_ceil))
        S_extend_L = np.zeros((M + L, M_ceil + L))

        # Forward direction
        if direction==0:
            S_extend_L[0:M,0:M_ceil] = S_resample
            for pos in range(0,L):
                S_L = S_L + S_extend_L[pos:(M + pos), pos:(M_ceil + pos)]

        # Backward direction
        if direction==1:
            S_extend_L[L:(M+L),L:(M_ceil+L)] = S_resample
            for pos in range(0,L):
                S_L = S_L + S_extend_L[(L-pos):(M + L - pos), (L-pos):(M_ceil + L - pos)]

        S_L = S_L/L
        resample = np.multiply(np.divide(np.arange(1,N+1),N),M_ceil)
        np.around(resample, 0, resample)
        resample = resample-1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)

        S_resample_inv = S_L[:, index_resample]
        S_L_final = np.maximum(S_L_final, S_resample_inv)

    return S_L_final

def shift_cyc_matrix(X, shift=0):
    """Cyclic shift of features matrix along first dimension

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X: Feature respresentation
        shift: Number of bins to be shifted

    Returns:
        X_cyc: Cyclically shifted feature matrix
    """
    #Note: X_cyc = np.roll(X, shift=shift, axis=0) does to work for jit
    K, N = X.shape
    shift = np.mod(shift, K)
    X_cyc = np.zeros((K,N))
    X_cyc[shift:K, :] = X[0:K-shift, :]
    X_cyc[0:shift, :] = X[K-shift:K, :]
    return X_cyc

def compute_SM_TI(X, Y, L=1, tempo_rel_set=[1], shift_set=[0], direction=2):
    """Compute enhanced similaity matrix by applying path smoothing and transpositions

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X, Y: Input feature sequences
        L: Length of filter
        tempo_rel_set: Set of relative tempo values
        shift_set: Set of shift indices
        direction: Direction of smoothing (0: forward; 1: backward; 2: both directions)

    Returns:
        S_TI: Transposition-invariant SM
        I_TI: Transposition index matrix
    """
    for shift in shift_set:
        X_cyc = shift_cyc_matrix(X, shift)
        S_cyc = compute_SM_dot(X,X_cyc)

        if direction==0:
            S_cyc = filter_diag_mult_SM(S_cyc, L, tempo_rel_set, direction=0)
        if direction==1:
            S_cyc = filter_diag_mult_SM(S_cyc, L, tempo_rel_set, direction=1)
        if direction==2:
            S_forward = filter_diag_mult_SM(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=0)
            S_backward = filter_diag_mult_SM(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=1)
            S_cyc = np.maximum(S_forward, S_backward)
        if shift ==  shift_set[0]:
            S_TI = S_cyc
            I_TI = np.ones((S_cyc.shape[0],S_cyc.shape[1])) * shift
        else:
            #jit does not like the following lines
            #I_greater = np.greater(S_cyc, S_TI)
            #I_greater = (S_cyc>S_TI)
            I_TI[S_cyc>S_TI] = shift
            S_TI = np.maximum(S_cyc, S_TI)

    return S_TI, I_TI

def compute_kernel_checkerboard_Gaussian(L, var=1, normalize=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1]
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L: Parameter specifying the kernel size M=2*L+1
        var: Variance parameter determing the tapering (epsilon)

    Returns:
        kernel: Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2)/(L*var)
    axis = np.arange(-L,L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D,gaussian1D)
    kernel_box = np.outer(np.sign(axis),np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel

def compute_novelty_SSM(S, kernel=None, L=10, var=0.5, exclude=False):
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S: SSM
        kernel: Checkerboard kernel (if kernel==None, it will be computed)
        L: Parameter specifying the kernel size M=2*L+1
        var: Variance parameter determing the tapering (epsilon)
        exclude: Sets the first L and last L values of novelty function to zero

    Returns:
        nov: Novelty function
    """    
    if kernel is None:
        kernel = compute_kernel_checkerboard_Gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    #np.pad does not work with numba/jit
    S_padded  = np.pad(S,L,mode='constant')
    
    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M]  * kernel)
    if exclude:
        right = np.min([L,N])
        left = np.max([0,N-L])
        nov[0:right] = 0
        nov[left:N] = 0
        
    return nov
    
def plot_matrix(X, Fs=1, Fs_F=1, T_coef=None, F_coef=None, xlabel='Time (seconds)', ylabel='Frequency (Hz)', title='',
                dpi=72, colorbar=True, colorbar_aspect=20.0, ax=None, figsize=(6, 3), **kwargs):
    """Plot a matrix, e.g. a spectrogram or a tempogram (function from Notebook: B/B_PythonVisualization.ipynb in [2])
 
    Args:
        X: The matrix
        Fs: Sample rate for axis 1
        Fs_F: Sample rate for axis 0
        T_coef: Time coeffients. If None, will be computed, based on Fs.
        F_coef: Frequency coeffients. If None, will be computed, based on Fs_F.
        xlabel: Label for x axis
        ylabel: Label for y axis
        title: Title for plot
        dpi: Dots per inch
        colorbar: Create a colorbar.
        colorbar_aspect: Aspect used for colorbar, in case only a single axes is used.
        ax: Either (1.) a list of two axes (first used for matrix, second for colorbar), or (2.) a list with a single
            axes (used for matrix), or (3.) None (an axes will be created).
        figsize: Width, height in inches
        **kwargs: Keyword arguments for matplotlib.pyplot.imshow

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        im: The image plot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = [ax]
    if T_coef is None:
        T_coef = np.arange(X.shape[1]) / Fs
    if F_coef is None:
        F_coef = np.arange(X.shape[0]) / Fs_F

    if 'extent' not in kwargs:
        x_ext1 = (T_coef[1] - T_coef[0]) / 2
        x_ext2 = (T_coef[-1] - T_coef[-2]) / 2
        y_ext1 = (F_coef[1] - F_coef[0]) / 2
        y_ext2 = (F_coef[-1] - F_coef[-2]) / 2
        kwargs['extent'] = [T_coef[0] - x_ext1, T_coef[-1] + x_ext2, F_coef[0] - y_ext1, F_coef[-1] + y_ext2]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray_r'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'

    im = ax[0].imshow(X, **kwargs)

    if len(ax) == 2 and colorbar:
        plt.colorbar(im, cax=ax[1])
    elif len(ax) == 2 and not colorbar:
        ax[1].set_axis_off()
    elif len(ax) == 1 and colorbar:
        plt.sca(ax[0])
        plt.colorbar(im, aspect=colorbar_aspect)

    ax[0].set_xlabel(xlabel, fontsize=14)
    ax[0].set_ylabel(ylabel, fontsize=14)
    ax[0].set_title(title, fontsize=18)

    if fig is not None:
        plt.tight_layout()

    return fig, ax, im
    
def compute_local_average(x, M, Fs=1):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        M: Determines size (2M+1*Fs) of local average
        Fs: Sampling rate

    Returns:
        local_average: Local average signal
    """
    L = len(x)
    M = int(np.ceil(M * Fs))
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

def spectral_flux(x, Fs=1, N=1024, W=640, H=80, gamma=100, M=20, norm=1, band=[]):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        M: Size (frames) of local average
        norm: Apply max norm (if norm==1)
        band: List of lower and upper spectral freq limits

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=W, window='hanning')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
	
    if len(band)!=0: 
        band = np.array(band)*(N/2+1)/Fs
        Y = Y[int(band[0]):int(band[1]),:]

    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm == 1:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum

'''
References
[1] Meinard Müller and Frank Zalkow: FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing. Proceedings of the International Conference on Music Information Retrieval (ISMIR), Delft, The Netherlands, 2019.
'''
