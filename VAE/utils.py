import matplotlib.pyplot as plt
import librosa.display
import stempeg
from preprocess import stereo_to_mono


def plot_spectrogram(file_path):
    """plot the spectrogram of the file_path file.

    Args:
        file_path (str): path of the wav file
    """
    audio_data , rate = stempeg.read_stems(file_path)
    mono = stereo_to_mono(audio_data[0])
    X = librosa.stft(mono)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=rate, x_axis='time', y_axis='hz')