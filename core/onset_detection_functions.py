import numpy as np

def sigm(x):
    k = np.exp(-x)
    s = 1 / (1 + k)
    return s

def tanh(x):
    t = np.tanh(x)
    return t

def compute_hfc(X, freqs, f_low, f_hig):
    """
    Compute High-Frequency Content (HFC) and its difference (ΔHFC) for a selected frequency band.
    
    Parameters:
    X : np.ndarray
        Complex STFT matrix (shape: frequency bins x time frames).
    freqs : np.ndarray
        Frequency values corresponding to STFT bins.
    f_low : float
        Lower frequency limit of the selected band.
    f_hig : float
        Upper frequency limit of the selected band.
    
    Returns:
    HFC : np.ndarray
        High-Frequency Content for each time frame.
    delta_HFC : np.ndarray
        Difference of HFC over time (ΔHFC).
    """
    # Compute magnitude spectrogram

    # X = np.log2(X + 1)
    X = sigm(X)
    # X = tanh(X)
    magnitude = np.abs(X)
    
    # Get indices corresponding to the selected frequency band
    freq_indices = np.where((freqs >= f_low) & (freqs <= f_hig))[0]
    
    # Compute HFC by weighting the squared magnitudes by k^2
    # HFC = np.sum(magnitude[freq_indices, :] * (freq_indices[:, None] ** 2), axis=0)
    HFC = np.sum(magnitude[freq_indices, :] , axis=0)

    # Compute ΔHFC (difference over time)
    delta_HFC = np.zeros_like(HFC)
    delta_HFC[1:] = HFC[1:] - HFC[:-1]  # Shifted difference
    delta_HFC[0] = delta_HFC[1]

    delta_HFC = (delta_HFC + np.abs(delta_HFC))/2

    return np.log10(delta_HFC+1)    