# import numpy as np
import matplotlib.pyplot as plt
# from pathlib import Path
import seaborn as sns
import matplotlib as mpl
from matplotlib import rc

import numpy as np

def configure_plots():
    """
    Configures the plotting environment for scientific and publication-quality plots,
    ensuring black axes lines and lighter grid lines.
    """

    # Set Seaborn style with a white background and lighter dashed grid lines.
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.color': '0.7',       # Lighter gray grid lines (0.0=black, 1.0=white)
        'grid.linewidth': 0.5
    })
    sns.set_context("paper")  # Adjust context to "talk" or "poster" for presentations

    # Enable TeX mode for high-quality text rendering
    rc('text', usetex=False)
    rc('font', family='serif')

    # Configure high-resolution figures for publication quality
    plt.rcParams['figure.dpi'] = 300

    # Set font sizes for various plot elements
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 9

    # Configure legend aesthetics
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "black"
    mpl.rcParams['legend.fontsize'] = 6
    mpl.rcParams['legend.framealpha'] = 1  # Full opacity for better readability

    # Adjust figure size for clarity
    plt.rcParams['figure.figsize'] = [8, 6]  # Width, height in inches

    # Use a colorblind-friendly palette (tab10)
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=mpl.cm.tab10.colors)

    # Set error bars and line properties for better visibility
    mpl.rcParams['errorbar.capsize'] = 2
    mpl.rcParams['lines.linewidth'] = 2

    # Prevent potential errors with large plots
    mpl.rcParams['agg.path.chunksize'] = 10000

    # Save figures with high resolution
    mpl.rcParams['savefig.dpi'] = 300

    # -- NEW LINES FOR BLACK AXES/SPINES --
    mpl.rcParams['axes.edgecolor'] = 'black'   # Axes (spine) color
    mpl.rcParams['axes.linewidth'] = 1.0       # Axes (spine) thickness

    # Create an annotation style dictionary (optional)
    annotation_style = {
        'fontsize': 6,
        'fontfamily': 'serif',
        'color': 'black',
        'backgroundcolor': 'white'
    }

    return annotation_style 

# Used to plot spectrograms
def plot_spectrogram(title, tt, ff, Sxx, cmap, ax, Scale=True, clim=True, fmin=None, fmax=None):
    """
    Plot a spectrogram on a given axes and optionally add a colorbar.

    Parameters:
    - title (str): Title of the plot.
    - tt (array): Array of time bins for the spectrogram.
    - ff (array): Array of frequency bins for the spectrogram.
    - Sxx (array): Spectrogram of the signal.
    - cmap (str): Colormap for the spectrogram.
    - Scale (bool): If True, scale the spectrogram to decibel.
    - ax (matplotlib.axes.Axes): Axes object where the spectrogram is plotted.
    - fmin (float): Minimum frequency to display in kHz. If None, uses minimum of ff.
    - fmax (float): Maximum frequency to display in kHz. If None, uses maximum of ff.

    The color limits for the spectrogram are automatically set based on the 90th to 99th percentiles of the spectrogram values to better highlight relevant features.

    Returns:
    None
    """
    if Scale:
        Sxx = np.log10(Sxx + np.finfo(float).eps)  # Ensure no log(0) issue

    # Store the QuadMesh object returned by pcolormesh
    mesh = ax.pcolormesh(tt, ff/1000, Sxx, cmap=cmap, shading='gouraud')
    ax.set_title(title)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (kHz)')

    # Set frequency limits
    if fmin is None:
        fmin = ff.min() / 1000
    if fmax is None:
        fmax = ff.max() / 1000
    
    ax.set_ylim(fmin, fmax)

    if clim:
        # Calculate 90th and 99th percentiles for color limit scaling
        vmin = np.percentile(Sxx, 90)
        vmax = np.percentile(Sxx, 99.99)

        # Set the color limits for the colormap using the calculated percentiles
        mesh.set_clim(vmin, vmax)