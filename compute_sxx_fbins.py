"""
compute_sxx_fbins.py

This script computes and saves reduced-band STFT representations (Sxx_fb) 
from '.wav' audio files, processing each file in predefined mid-term temporal 
windows (e.g., 1-second segments). 

Main Steps:
-----------
1. Load WAV files from a specified input directory.
2. Divide each WAV file into mid-term windows of predefined duration (e.g., 1 second).
3. For each mid-term window:
   - Compute the Short-Time Fourier Transform (STFT) using a short-term analysis window 
     (e.g., 1 ms).
   - Aggregate the STFT into a reduced number of frequency bands using linear frequency 
     scaling or other user-defined band divisions.
4. Concatenate results across mid-term windows to form:
   - `Sxx_fb`: matrix of reduced-band STFT (frequency bands × time frames).
   - `freq_vector`: center frequencies of the aggregated bands.
   - `time_vector`: time stamps for each short-term frame covering the full file duration.
5. Save the computed data (Sxx_fb, freq_vector, and time_vector) in a suitable file format 
   (e.g., '.npz' or '.mat') for later analysis or visualization.

Expected Output per Audio File:
-------------------------------
- `Sxx_fb`: Matrix of reduced-band STFT (frequency bands × time frames).
- `freq_vector`: Array of center frequencies for each aggregated band.
- `time_vector`: Array of time stamps corresponding to each short-term frame.

Notes:
------
- The script is optimized for handling long recordings by splitting them into manageable 
  mid-term windows, ensuring memory efficiency.
- The frequency bands can be defined linearly to allow equally spaced divisions across 
  the frequency range of interest.

"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from scipy.signal import  ShortTimeFFT, windows, spectrogram
from librosa.feature import melspectrogram

import core.signal_processing as sp

from soundfile import read, info

#### Local functions #########
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
    return file_info.samplerate, file_info.duration

def compute_Sxx_fband(file_path, mid_term_sec, short_term_sec, num_fbands):
    """
    Compute reduced frequency band representation (Sxx_fband) from STFT using pre-defined
    mid-term and short-term window sizes. Aggregates frequency bins into Mel or linear bands.

    Args:
        file_path (str): Path to the audio file to process.
        mid_term_sec (float): Duration of mid-term window in seconds (e.g., 30 s).
        short_term_sec (float): Duration of short-term window in seconds (e.g., 1 s).
        num_fbands (int): Number of frequency bands to aggregate into.

    Returns:
        tuple: (Sxx_fband, freqs_fband, time_fr)
            - Sxx_fband (np.ndarray): (num_fbands, expected_frames), aggregated spectral content.
            - freqs_fband (np.ndarray): Center frequencies of each band (Mel scale in this version).
            - time_fr (np.ndarray): Array of time stamps for each frame (in seconds).
    """

    # 1. Get sampling rate and duration
    fs, duration = get_sample_rate(file_path)

    # 2. Expected number of frames for 120 seconds of recording
    expected_frames = int(np.ceil(120 / short_term_sec))

    # 3. Pre-allocate result arrays (initialized as NaN)
    Sxx_fband = np.full((num_fbands, expected_frames), np.nan)
    time_fr = np.full(expected_frames, np.nan)

    # 4. Mid-term window parameters
    num_samples_mid = int(mid_term_sec * fs)
    start_sample = 0
    frame_counter = 0  # Total number of processed frames

    # 5. Placeholder for frequency bands (computed during first window)
    freqs_fband = None

    # === Loop over mid-term windows === #
    while True:
        # 6. Read mid-term window
        sound_data = sp.read_audio_segment(file_path, start_sample, num_samples_mid)
        if sound_data is None or len(sound_data) == 0:
            break  # Stop if no more data

        # 7. Compute STFT using short-term resolution
        W = int(short_term_sec * fs)
        nfft = W
        freqs, times, Sxx = spectrogram(
            sound_data,
            fs=fs,
            nperseg=W,
            noverlap=0,
            nfft=nfft,
            scaling='density',
            mode='magnitude'
        )

        # 8. Aggregate STFT into frequency bands
        fmin_mel = 100
        fmax_mel = fs/2
        Sxx_band_chunk = melspectrogram(S=Sxx, sr=fs, fmin=fmin_mel, fmax=fmax_mel, n_mels=num_fbands)
        
        # 9. Set up frequency bands if first iteration
        if freqs_fband is None:
            freqs_fband = librosa.mel_frequencies(n_mels=num_fbands, fmin=fmin_mel, fmax=fmax_mel)

        # 10. Generate absolute time for current frames
        frames_in_chunk = Sxx_band_chunk.shape[1]
        abs_times_chunk = (start_sample / fs) + np.arange(frames_in_chunk) * short_term_sec

        # 11. Handle possible overflow if chunk exceeds expected total frames
        end_frame = frame_counter + frames_in_chunk
        if end_frame > expected_frames:
            frames_in_chunk = expected_frames - frame_counter
            end_frame = expected_frames
            Sxx_band_chunk = Sxx_band_chunk[:, :frames_in_chunk]
            abs_times_chunk = abs_times_chunk[:frames_in_chunk]

        # 12. Store computed chunk
        Sxx_fband[:, frame_counter:end_frame] = Sxx_band_chunk
        time_fr[frame_counter:end_frame] = abs_times_chunk

        # 13. Update frame counter and next window start
        frame_counter = end_frame
        start_sample += num_samples_mid

    # 14. Handle incomplete files: replicate last frame if needed
    if frame_counter < expected_frames and frame_counter > 0:
        # Extract last valid frame and time
        last_frame = Sxx_fband[:, frame_counter - 1][:, np.newaxis]
        last_time = time_fr[frame_counter - 1]

        frames_to_fill = expected_frames - frame_counter
        # Replicate last valid frame and time
        Sxx_fband[:, frame_counter:] = np.repeat(last_frame, frames_to_fill, axis=1)
        time_fr[frame_counter:] = last_time + np.arange(1, frames_to_fill + 1) * short_term_sec

    # If no frame was computed (empty file), return NaNs as initialized

    return Sxx_fband, freqs_fband, time_fr

#### Main function definition ########
def main(input_base_dir, output_base_dir):
    """
    Main entry point for the Sxx frequency band computation script.

    Workflow:
    1. Iterates over all WAV files located inside folders following the "2024*" pattern in `input_base_dir`.
    2. For each WAV file:
        - Calls `compute_Sxx_fband()` to compute frequency-banded STFT (Sxx_fband).
        - Receives: 
            * Sxx_fband: STFT aggregated in frequency bands (shape: num_bands x time_frames).
            * freqs_fband: Center frequencies of each frequency band.
            * time_fr: Time vector associated with each frame (in seconds).
    3. Saves the results for each file as a compressed `.npz` file under `output_base_dir`,
       preserving the folder structure and using the same filename stem.
    
    Args:
        input_base_dir (str or Path): Path to the base directory containing the WAV files.
        output_base_dir (str or Path): Path to the output directory where the Sxx_fband results will be stored.
    """

    # Convert input directories to Path objects for safety
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)

    # Parameters for segmentation and frequency banding
    mid_term_sec = 30  # Mid-term window length in seconds (e.g., for memory management)
    short_term_sec = 0.1  # Short-term window length in seconds (used for STFT)
    num_fbands = 80#300  # Number of linear frequency bands (or mel freq bands) for aggregation

    # Iterate over subfolders that match the "2024*" pattern (recording dates)
    for folder in input_base_dir.glob("2024*"):
        # Iterate over each .wav file within the folder
        for audio_file in folder.glob("*.wav"):
            print(f"Processing {audio_file} ...")
            # Clear the terminal screen
            os.system('cls' if os.name == 'nt' else 'clear')

            # Call the function to compute Sxx reduced in frequency bands
            Sxx_fband, freqs_fband, time_fr = compute_Sxx_fband(
                audio_file, mid_term_sec, short_term_sec, num_fbands
            )
            # Sxx_fband, freqs_fband, time_fr = [0,0,0]

            # Define the output path (mirror input folder structure)
            output_file = output_base_dir / folder.name / f"{audio_file.stem}.npz"

            # Ensure that the output folder exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save all results in compressed NumPy format (.npz)
            np.savez(
                output_file,
                Sxx_fband=Sxx_fband,
                freqs_fband=freqs_fband,
                time_frame=time_fr
            )
            print(f"Saved output to: {output_file}")


#### Call main function ##########
if __name__ == "__main__":
    # Define base input/output directories (can be changed for testing)
    input_base_dir = "../data/wav-files"
    output_base_dir = "../data/mel_Sxx_fband"

    # Run the main function
    main(input_base_dir, output_base_dir)