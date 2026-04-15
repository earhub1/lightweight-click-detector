# signal_processing.py

from soundfile import read, info
import numpy as np

def get_sample_rate(file_path):
    """
    Retrieve the sample rate from an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        int: The sample rate of the audio file (samples per second).
    """
    # Get audio file information to extract the sample rate
    file_info = info(file_path, verbose=False)
    file_info.endian
    return file_info.samplerate

def read_audio_segment(file_path, start_sample, num_samples):
    """
    Reads a specified segment from an audio file based on sample indices.

    Args:
        file_path (str): The path to the audio file.
        start_sample (int): The starting sample index to read from. This value determines the starting point
                            within the audio file from which the segment will be read.
        num_samples (int): The number of samples to read from the audio file starting from `start_sample`.
                           This specifies the duration of the segment in terms of the number of samples.

    Returns:
        numpy.ndarray: The audio samples of the specified segment as a 1D array. If the audio file is stereo,
                       only the first channel is returned as a mono signal. Returns None if the specified
                       segment exceeds the bounds of the available audio data (i.e., if `start_sample + num_samples`
                       is greater than the total number of samples in the audio file).

    Example:
        To read a 5-second segment starting 10 seconds into the audio file with a sample rate of 44100 Hz:

        >>> file_path = 'path/to/audio.wav'
        >>> start_sample = int(10 * 44100)  # Start at 10 seconds
        >>> num_samples = int(5 * 44100)    # Read 5 seconds worth of samples
        >>> segment = read_audio_segment(file_path, start_sample, num_samples)
        >>> if segment is not None:
        >>>     print(f"Segment length: {len(segment)} samples")
        >>> else:
        >>>     print("Requested segment is out of bounds.")
    """
    # Get audio file information to extract the sample rate
    file_info = info(file_path, verbose=False)
    total_samples = file_info.frames

    # Check if the requested segment is within the valid range of the audio file
    if total_samples < start_sample + num_samples:
        return None

    # Read the desired segment from the file
    data, _ = read(file_path, start=start_sample, frames=num_samples, always_2d=True)

    # Extract the first channel if the audio is stereo
    data = data[:, 0]  
    
    return data

def aggregate_stft_linear_bands(Sxx, freqs, num_bands):
    """
    Aggregate STFT data (Sxx) into a reduced number of linear frequency bands, and compute 
    the center frequency of each band.

    This function groups STFT frequency bins into equally spaced frequency bands, 
    summing (or averaging) the energy content within each band to create a low-dimensional 
    representation of the STFT. It also returns the center frequency of each band 
    for interpretation and plotting.

    Args:
    ----
    Sxx : np.ndarray
        2D array of STFT magnitude or power (shape: freq_bins x time_frames).
    freqs : np.ndarray
        1D array of frequency values (Hz) corresponding to STFT frequency bins (length: freq_bins).
    num_bands : int
        Number of linear frequency bands to aggregate the STFT into.

    Returns:
    --------
    aggregated : np.ndarray
        Aggregated STFT matrix of shape (num_bands x time_frames), where each row 
        corresponds to one linear frequency band.
    freqs_bin : list of float
        List containing the center frequency (Hz) of each aggregated band, computed 
        as the mean of the band's lower and upper edges.

    Example:
    --------
    >>> aggregated, freqs_center = aggregate_stft_linear_bands(Sxx, freqs, num_bands=10)
    >>> print("Aggregated shape:", aggregated.shape)
    >>> print("Center frequencies:", freqs_center)
    """
    num_freq_bins, num_frames = Sxx.shape

    # Step 1: Define frequency edges for the bands
    f_min, f_max = freqs[0], freqs[-1]
    band_edges = np.linspace(f_min, f_max, num_bands + 1)  # Define edges (num_bands + 1 points)

    # Step 2: Initialize the aggregated array
    aggregated = np.zeros((num_bands, num_frames))

    # Step 3: Aggregate energy in each band
    for i in range(num_bands):
        # Define band edges
        f_low, f_high = band_edges[i], band_edges[i + 1]

        # Find indices of frequency bins within the current band
        idx = np.where((freqs >= f_low) & (freqs < f_high))[0]

        if len(idx) > 0:
            # Sum energy across the selected frequency bins
            aggregated[i, :] = np.sum(Sxx[idx, :], axis=0)
        else:
            # If no frequency bin falls in this range, fill with NaNs
            aggregated[i, :] = np.nan

    # Step 4: Prepare list of center frequencies for each band
    freqs_bin = [np.mean([band_edges[i], band_edges[i + 1]]) for i in range(num_bands)]

    return aggregated, freqs_bin


def compute_cd(X, method="phase_init"):
    """
    Compute Complex Domain (CD) Onset Detection function from the one-sided STFT.

    Parameters:
    X : np.ndarray
        One-sided complex STFT matrix (shape: frequency bins x time frames).
    method : str, optional
        Padding method for phase differences. Options:
        - "phase_init" (default): Use phase initialization padding.
        - "zero_pad": Use explicit zero-padding.

    Returns:
    CD : np.ndarray
        Complex Domain Onset Detection feature over time.
    """
    # Extract phase from STFT
    phi = np.angle(X)  # Ensures phase values are within [-π, π]
    
    # First-order phase difference
    phi_prime = np.zeros_like(phi)  # Initialize
    phi_prime[:, 1:] = phi[:, 1:] - phi[:, :-1]  # Compute phase difference

    # Ensure phase remains in [-π, π] after subtraction
    phi_prime = np.mod(phi_prime + np.pi, 2 * np.pi) - np.pi

    # Initialize first values based on method
    if method == "phase_init":
        phi_prime[:, 0] = phi_prime[:, 1]  # Approximate first frame
    elif method == "zero_pad":
        pass  # Already initialized as zeros

    # Compute target complex STFT: X_T(n,k) = |X(n,k)| * exp(j(ϕ'(n,k) + ϕ'(n-1,k)))
    phi_sum = np.zeros_like(phi)
    phi_sum[:, 1:] = phi_prime[:, 1:] + phi_prime[:, :-1]  # Sum of phase differences
    
    if method == "phase_init":
        phi_sum[:, 0] = phi_prime[:, 0]  # Ensure consistency in first frame

    # Compute X_T
    X_T = np.abs(X) * np.exp(1j * phi_sum)

    X1 = sigm(np.abs(X)) * np.exp(1j * phi)

    # Compute Complex Domain Onset Detection
    CD = np.sum(tanh(np.abs(X - X_T)), axis=0)
    # CD = np.sum((np.abs(np.exp(1j * phi) - np.exp(1j * phi_sum))), axis=0)

    return np.log10(CD + 1)

def compute_wpd(X, method="phase_init"):
    """
    Compute Weighted Phase Deviation (WPD) from the one-sided STFT.

    Parameters:
    X : np.ndarray
        One-sided complex STFT matrix (shape: frequency bins x time frames).
    method : str, optional
        Padding method for phase differences. Options:
        - "phase_init" (default): Use phase initialization padding.
        - "zero_pad": Use explicit zero-padding.

    Returns:
    WPD : np.ndarray
        Weighted Phase Deviation feature over time.
    """
    # Extract phase from STFT
    phi = np.angle(X)  # Ensures phase values are within [-π, π]

    # First-order phase difference
    phi_prime = np.zeros_like(phi)  # Initialize
    phi_prime[:, 1:] = phi[:, 1:] - phi[:, :-1]  # Compute phase difference

    # Ensure phase remains in [-π, π] after subtraction
    phi_prime = np.mod(phi_prime + np.pi, 2 * np.pi) - np.pi

    # Second-order phase difference
    phi_2prime = np.zeros_like(phi)

    if method == "phase_init":
        # Use phase initialization padding
        phi_prime[:, 0] = phi_prime[:, 1]  # Approximate first difference using second value
        phi_2prime[:, 1:] = phi_prime[:, 1:] - phi_prime[:, :-1]  # Compute second-order difference
        phi_2prime[:, 0] = phi_2prime[:, 1]  # Ensure continuity by approximating the first value
    elif method == "zero_pad":
        # Default zero-padding is already set in phi_2prime
        phi_2prime[:, 1:] = phi_prime[:, 1:] - phi_prime[:, :-1]  # Compute second-order difference

    X1 = sigm(np.abs(X))
    # Compute Weighted Phase Deviation (WPD)
    WPD = (2 / X.shape[0]) * np.sum(sigm(np.abs(X*phi_2prime)), axis=0)  # Ensure temporal consistency

    return np.log2(WPD + 1)

def compute_spectral_flux(X, freqs, f_low, f_high):
    """
    Compute Spectral Flux (SF) for a selected frequency band.
    
    Parameters:
    X : np.ndarray
        Magnitude spectrogram (one-sided STFT, shape: frequency bins x time frames).
    freqs : np.ndarray
        Frequency vector corresponding to STFT bins.
    f_low : float
        Lower frequency limit of the selected band.
    f_high : float
        Upper frequency limit of the selected band.
    
    Returns:
    SF : np.ndarray
        Spectral Flux values over time.
    """
    # Get indices for the selected frequency band
    freq_indices = np.where((freqs >= f_low) & (freqs <= f_high))[0]

    X = sigm(X)

    # Compute spectral magnitude differences across time
    delta_X = np.zeros_like(X[freq_indices, :])
    delta_X[:, 1:] = np.abs(X[freq_indices, 1:]) - np.abs(X[freq_indices, :-1])
    
    # Ensure the first frame maintains temporal dimensions by copying the second frame
    delta_X[:, 0] = delta_X[:, 1]

    # Apply half-wave rectification: H(x) = (x + |x|) / 2
    H = (delta_X + np.abs(delta_X)) / 2

    # Sum over frequency bins to obtain SF
    SF = np.sum(H, axis=0)

    return np.log10(SF+1)

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

def sigm(x):
    k = np.exp(-x)
    s = 1 / (1 + k)
    return s

def tanh(x):
    t = np.tanh(x)
    return t