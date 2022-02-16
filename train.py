import torch
import argparse
from torch import optim
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from asteroid.models import ConvTasNet
from asteroid.data import MUSDB18Dataset
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper


#####################
##### CONSTANTS #####
#####################
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", default="musdb_data", type=str)
parser.add_argument("--size", default=None, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument("--ckpdir", default="weights", type=str)
args = parser.parse_args()

SAMPLE_RATE = 44100
DATA_DIR = Path(args.datadir)
SIZE = args.size
LR = args.lr
N_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
CKP_DIR = Path(args.ckpdir)
CKP_PATH = CKP_DIR/f"model_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth"

if not CKP_DIR.exists():
    CKP_DIR.mkdir(parents=True)


################
##### DATA #####
################
train_dataset = MUSDB18Dataset(
    root=DATA_DIR.__str__(),
    targets=["vocals", "bass", "drums", "other"],
    suffix=".mp4",
    split="train",
    subset=None,
    segment=1,
    samples_per_track=1,
    random_segments=True,
    random_track_mix=False,
    sample_rate=SAMPLE_RATE,
    size=SIZE
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
print(">>> Training Dataloader ready")

test_dataset = MUSDB18Dataset(
    root=DATA_DIR.__str__(),
    targets=["vocals", "bass", "drums", "other"],
    suffix=".mp4",
    split="test",
    subset=None,
    segment=1,
    samples_per_track=1,
    random_segments=True,
    random_track_mix=False,
    sample_rate=SAMPLE_RATE,
    size=SIZE
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print(">>> TEST Dataloader ready")


################
##### MODEL ####
################
model = ConvTasNet(n_src=4, sample_rate=SAMPLE_RATE)
loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
optimizer = optim.Adam(model.parameters(), lr=LR)


################
### TRAINING ###
################
def train(model, dataset, criterion, optim, epochs):
    print("\n>>> Begin training\n")
    
    for epoch in range(epochs):
        print(">>> EPOCH", epoch)
        epoch_loss = 0
        data_counter = 0
        
        for n_batch, train_batch in enumerate(dataset):
            optim.zero_grad()

            x, y = train_batch
            
            output = model(x)

            loss = criterion(output, y)
            epoch_loss += loss.item()
            
            batch_size = x.shape[0]
            data_counter += batch_size

            loss.backward()
            optimizer.step()
            # print(f"Batch {n_batch} loss = {loss.item() / batch_size}")
            
        epoch_loss /= data_counter
        print(f"Epoch {epoch} - Mean Loss: {epoch_loss}")

train(model, train_loader, loss, optimizer, N_EPOCHS)

################
### Save #######
################
torch.save(model, CKP_PATH)