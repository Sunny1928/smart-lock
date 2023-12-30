import librosa
import numpy as np
import soundfile as sf
import myconfig


def extract_features(audio_file):
    """Extract MFCC features from an audio file, shape=(TIME, MFCC)."""
    waveform, sample_rate = sf.read(audio_file)

    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr = sample_rate, target_sr = 16000)

    # Mel-frequency cepstral coefficients (MFCCs) are robust to noise bcoz of logarithmic compression
    features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=myconfig.N_MFCC)
    # the shape of features will be 40 X 441, where 40 represent featues where as 441 represent frames

    return features.transpose()

def extract_sliding_windows(features):
    """Extract sliding windows from features."""
    sliding_windows = []
    start = 0
    while start + myconfig.SEQ_LEN <= features.shape[0]:
        sliding_windows.append(features[start: start + myconfig.SEQ_LEN, :])
        start += myconfig.SLIDING_WINDOW_STEP
    return sliding_windows