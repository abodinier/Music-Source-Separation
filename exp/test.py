import torch
import stempeg
import librosa
from torch import optim
from scipy.io.wavfile import write

from pytorch_lightning import Trainer

from asteroid.models import ConvTasNet
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper

from asteroid.data import MUSDB18Dataset

from asteroid.engine import System

from pathlib import Path
from IPython.display import display, Audio


SAMPLE_RATE = 8000


def prepare_musdb18(source, dest, extension="mp4", n=-1, sr=44100):
    count = 0
    names = ("mixture", "drums", "bass", "other", "vocals")

    for f in source.glob(f"*.{extension}"):
        song = str(f.stem).split(".")[0]
        
        if (dest/song).is_dir():  # Skip existing WAVs
            continue

        s, rate = stempeg.read_stems(f.__str__())

        (dest/song).mkdir(parents=True)

        for name, wav in zip(names, s):
            write(dest/song/f"{name}.wav", sr, librosa.to_mono(wav.T))

        count += 1
        if n > 0 and count >= n:
            break

    print(f"{'=' * 5} Preparation completed ({count} songs) {'=' * 5}")


train_loader = MUSDB18Dataset(
    root="/content/drive/MyDrive/IA321/data/musdb18/prep",
    sources=["vocals", "bass", "drums", "other"],
    targets=["vocals", "bass", "drums", "other"],
    suffix=".wav",
    split="train",
    subset=None,
    segment=1,
    samples_per_track=1,
    random_segments=True,
    random_track_mix=False,
    sample_rate=SAMPLE_RATE
)

val_loader = MUSDB18Dataset(
    root="/content/drive/MyDrive/IA321/data/musdb18/prep",
    sources=["vocals", "bass", "drums", "other"],
    targets=["vocals", "bass", "drums", "other"],
    suffix=".wav",
    split="test",
    subset=None,
    segment=1,
    samples_per_track=1,
    random_segments=True,
    random_track_mix=False,
    sample_rate=SAMPLE_RATE
)


model = ConvTasNet(n_src=4, sample_rate=SAMPLE_RATE)

loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(model, dataset, criterion, optim, epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        for train_batch in dataset:
            optim.zero_grad()

            x, y = train_batch
            n_sources, n_batch, n_samples = y.size()
            y = y.reshape(n_batch, n_sources, n_samples)
            
            output = model(x)

            loss = criterion(output, y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} - Loss: {epoch_loss}")

train(model, train_loader, loss, optimizer, 1)

path = "/content/drive/MyDrive/IA321/data/musdb18/prep/test/AM Contra - Heart Peripheral/mixture.wav"
original = librosa.load(path, sr=SAMPLE_RATE)
output = model.separate(path)

