import yaml
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from torch import optim
from pathlib import Path
from datetime import datetime
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from asteroid.models import ConvTasNet
from asteroid.data import MUSDB18Dataset
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper


#####################
##### ARGS ##########
#####################
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_path", default="cfg.yaml", type=str)
parser.add_argument("--data_dir", default="musdb_data", type=str)
parser.add_argument("--ckpdir", default="weights", type=str)
parser.add_argument("--restore", default=None, type=str)
args = parser.parse_args()

CKP_PATH = Path(args.ckpdir)/f"training_{datetime.now().strftime('%Y%m%d-%H%M%S')}" if args.restore is None else Path(args.restore)
CKP_PATH_MODEL = CKP_PATH/"model.pth"
CKP_PATH_HISTORY = CKP_PATH/"history.csv"
CKP_PATH_CFG = CKP_PATH/"cfg.yaml"

if not CKP_PATH.is_dir():
    CKP_PATH.mkdir(parents=True)

if not CKP_PATH_CFG.exists():
    shutil.copy(args.cfg_path, CKP_PATH_CFG)

with open(str(CKP_PATH_CFG), 'r') as file:
    CFG = yaml.load(file, Loader=yaml.FullLoader)

DATA_DIR = Path(args.data_dir)
SEGMENT_SIZE = CFG["segment_size"]
RANDOM_TRACK_MIX = CFG["random_track_mix"]
TARGETS = CFG["targets"]
N_SRC = len(TARGETS)

#####################
##### HYPER-PARAMETERS
#####################
SAMPLE_RATE = CFG["sample_rate"]
SIZE = CFG["size"]
LR = CFG["learning_rate"]
N_EPOCHS = CFG["n_epochs"]
BATCH_SIZE = CFG["batch_size"]

N_BLOCKS = CFG["n_blocks"]
N_REPEATS = CFG["n_repeats"]
BN_CHAN = CFG["bn_chan"]
HID_CHAN = CFG["hid_chan"]
SKIP_CHAN = CFG["skip_chan"]
CONV_KERNEL_SIZE = CFG["conv_kernel_size"]
KERNEL_SIZE = CFG["kernel_size"]
N_FILTERS = CFG["n_filters"]
STRIDE = CFG["stride"]


if not CKP_PATH.exists():
    CKP_PATH.mkdir(parents=True)


################
##### DATA #####
################
train_dataset = MUSDB18Dataset(
    root=DATA_DIR.__str__(),
    targets=TARGETS,
    suffix=".mp4",
    split="train",
    subset=None,
    segment=SEGMENT_SIZE,
    samples_per_track=1,
    random_segments=True,
    random_track_mix=RANDOM_TRACK_MIX,
    sample_rate=SAMPLE_RATE,
    size=SIZE
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
print(">>> Training Dataloader ready")

test_dataset = MUSDB18Dataset(
    root=DATA_DIR.__str__(),
    targets=TARGETS,
    suffix=".mp4",
    split="test",
    subset=None,
    segment=SEGMENT_SIZE,
    samples_per_track=1,
    random_segments=True,
    random_track_mix=RANDOM_TRACK_MIX,
    sample_rate=SAMPLE_RATE,
    size=SIZE
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print(">>> TEST Dataloader ready")


################
##### MODEL ####
################
model = ConvTasNet(
    n_src=N_SRC,
    sample_rate=SAMPLE_RATE,
    n_blocks=N_BLOCKS,
    n_repeats=N_REPEATS,
    bn_chan=BN_CHAN,
    hid_chan=HID_CHAN,
    skip_chan=SKIP_CHAN,
    conv_kernel_size=CONV_KERNEL_SIZE,
    norm_type="gLN",
    mask_act="sigmoid",
    kernel_size=KERNEL_SIZE,
    n_filters=N_FILTERS,
    stride=STRIDE)

loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_updater = lr_scheduler.StepLR(optimizer, 20, 1e-2)
history = None


if args.restore is not None:
    model.load_state_dict(torch.load(CKP_PATH_MODEL)["model_state_dict"])
    optimizer.load_state_dict(torch.load(CKP_PATH_MODEL)["optimizer_state_dict"])
    lr_updater.load_state_dict(torch.load(CKP_PATH_MODEL)["lr_scheduler"])
    history = pd.read_csv(CKP_PATH_HISTORY)


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