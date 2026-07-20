import numpy as np
import matplotlib.pyplot as plt

from librosa.display import waveshow, specshow

def plot_audio(y: np.ndarray, sr: float, ax: plt.Axes = None):
    """Plot an audio signal

    Args:
        y (np.ndarray): the audio signal
        sr (float): the sampling rate at which audio signal was sampled
        ax (plt.Axes, optional): if provided, the axis to plot on. Defaults to None.
    """
    if ax is None:
        _, ax = plt.subplots()
    waveshow(y, sr=sr, ax=ax)

def plot_features(X: np.ndarray, Y: np.ndarray, hop_length: int, sr: float, titles=('X', 'Y')):
    """Plot two chroma-like feature matrices side-by-side."""
    fig, ax = plt.subplots(nrows=2, sharex=False, figsize=(10, 5))
    img = specshow(X, x_axis='frames', y_axis='chroma',
                             hop_length=hop_length, cmap='gray_r', ax=ax[0])
    specshow(Y, x_axis='frames', y_axis='chroma',
                             hop_length=hop_length, cmap='gray_r', ax=ax[1])
    ax[0].set_title(titles[0])
    ax[1].set_title(titles[1])
    plt.tight_layout()
    plt.colorbar(img, ax=ax)

    return fig, ax

def plot_matrix(X, Fs=1, Fs_F=1, T_coef=None, F_coef=None,
                xlabel='Time (s)', ylabel='Frequency (Hz)',
                xlim=None, ylim=None, clim=None, title='',
                colorbar=True, ax=None, figsize=(6, 3), **kwargs):
    """2D raster visualization of a matrix, e.g. a spectrogram or a tempogram.

    Args:
        X: The matrix
        Fs: Sample rate for axis 1 (Default value = 1)
        Fs_F: Sample rate for axis 0 (Default value = 1)
        T_coef: Time coeffients. If None, will be computed, based on Fs. (Default value = None)
        F_coef: Frequency coeffients. If None, will be computed, based on Fs_F. (Default value = None)
        xlabel: Label for x-axis (Default value = 'Time (seconds)')
        ylabel: Label for y-axis (Default value = 'Frequency (Hz)')
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for y-axis (Default value = None)
        clim: Color limits (Default value = None)
        title: Title for plot (Default value = '')
        colorbar: Create a colorbar. (Default value = True)
        ax: Either (1.) a list of two axes (first used for matrix, second for colorbar), or (2.) a list with a single
            axes (used for matrix), or (3.) None (an axes will be created). (Default value = None)
        figsize: Width, height in inches (Default value = (6, 3))
        **kwargs: Keyword arguments for matplotlib.pyplot.imshow

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        im: The image plot

    ---
    Source : 
    This function comes from the library `libfmp`, part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP) 
    Link : https://github.com/meinardmueller/libfmp
    Licence : The MIT license
    """
    fig = None
    if ax is None:
        fig, axx = plt.subplots(1, 1, figsize=figsize)
        ax = [axx]
    if T_coef is None:
        T_coef = np.arange(X.shape[1]) / Fs
    if F_coef is None:
        F_coef = np.arange(X.shape[0]) / Fs_F
    kwargs.setdefault('cmap', 'gray_r')
    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('origin', 'lower')
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('extent', [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]])
    im = ax[0].imshow(X, **kwargs)
    if colorbar:
        plt.colorbar(im, ax=ax[0])
    ax[0].set(xlabel=xlabel, ylabel=ylabel, title=title)
    if xlim: ax[0].set_xlim(xlim)
    if ylim: ax[0].set_ylim(ylim)
    if clim: im.set_clim(clim)
    return fig, ax, im


def plot_alignment_path(C, path, title='Cost matrix with warping path',
                        xlabel='Y frames', ylabel='X frames', ax=None):
    """Compute the cost matrix of two feature sequences

    Args:
        C: Matrix to be plotted
        path: List of index pairs, to be visualized on the matrix (Default value = np.empty((0, 2)))
        **kwargs: keyword arguments for :func:`libfmp.b.b_plot.plot_matrix`

    Returns:
        fig: Handle for figure
        im: Handle for imshow
        line: handle for line plot

    ---
    Source : 
    This function comes from the library `libfmp`, part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP) 
    Link : https://github.com/meinardmueller/libfmp
    Licence : The MIT license
    """
    fig, ax, im = plot_matrix(C, ax=[ax] if ax is not None else None,
                              title=title, xlabel=xlabel, ylabel=ylabel,
                              clim=[0, np.max(C)])
    line = ax[0].plot(path[:, 1], path[:, 0], color='r', linewidth=1.5)

    if fig is not None:
        plt.tight_layout()

    return fig, im, line


def plot_alignment_connections(X, Y, path, hop_length, step=5):
    """Plot both feature sequences with colored lines connecting aligned frames."""
    N, M = X.shape[1], Y.shape[1]
    fig = plt.figure(figsize=(10, 4))
    ax_X = plt.axes([0, 0.60, 1, 0.40])
    specshow(X, ax=ax_X, x_axis='frames', y_axis='chroma',
                             hop_length=hop_length, cmap='gray_r')
    ax_X.set_ylabel('X'); ax_X.xaxis.tick_top()
    ax_Y = plt.axes([0, 0, 1, 0.40])
    specshow(Y, ax=ax_Y, x_axis='frames', y_axis='chroma',
                             hop_length=hop_length, cmap='gray_r')
    ax_Y.set_ylabel('Y')
    ax_mid = plt.axes([0, 0.40, 1, 0.20])
    for p in path[0:-1:step]:
        ax_X.axvline(p[0], color='r', alpha=0.5)
        ax_Y.axvline(p[1], color='r', alpha=0.5)
        ax_mid.plot((p[0]/N, p[1]/M), (1, -1), color='r')
    ax_mid.set(xlim=(0, 1), ylim=(-1, 1), xticks=[], yticks=[])
    return fig