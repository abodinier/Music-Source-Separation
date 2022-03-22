import matplotlib.pyplot as plt
import librosa.display
import stempeg



def plot_spectrogram(file_path):
    audio_data , rate = stempeg.read_stems(filelist_train[5])
    mono = stereo_to_mono(audio_data[0])
    X = librosa.stft(mono)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=rate, x_axis='time', y_axis='hz')