from dataclasses import dataclass
import yaml
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from torch import ne, optim
from pathlib import Path
from datetime import datetime
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from conv_tasnet import TasNet
from utility import sdr

from asteroid.data import MUSDB18Dataset
from asteroid.losses import pairwise_mse, PITLossWrapper
from asteroid.losses import pairwise_neg_sisdr

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
CKP_LOGS = CKP_PATH/"logs"
CKP_PATH_MODEL = CKP_PATH/"model.pth"
CKP_PATH_HISTORY = CKP_PATH/"history.csv"
CKP_PATH_CFG = CKP_PATH/"cfg.yaml"

if not CKP_PATH.is_dir():
    CKP_PATH.mkdir(parents=True)

if not CKP_LOGS.is_dir():
    CKP_LOGS.mkdir(parents=True)

if not CKP_PATH_CFG.exists():
    shutil.copy(args.cfg_path, CKP_PATH_CFG)

with open(str(CKP_PATH_CFG), 'r') as file:
    CFG = yaml.load(file, Loader=yaml.FullLoader)

DATA_DIR = Path(args.data_dir)
SEGMENT_SIZE = CFG["segment_size"]
RANDOM_TRACK_MIX = CFG["random_track_mix"]
TARGETS = CFG["targets"]
N_SRC = len(TARGETS)
LOSS = eval(CFG["loss"])
STORE_GRADIENT_NORM = CFG["store_gradient_norm"]

#####################
##### HYPER-PARAMETERS
#####################
SAMPLE_RATE = CFG["sample_rate"]
SIZE = None if CFG["size"] == -1 else CFG["size"]
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
model = TasNet(
    enc_dim=N_FILTERS,
    feature_dim=BN_CHAN,
    sr=SAMPLE_RATE,
    win=SEGMENT_SIZE,
    layer=N_BLOCKS,
    stack=N_REPEATS,
    kernel=CONV_KERNEL_SIZE,
    num_spk=N_SRC,
    causal=False
)

loss = sdr.batch_SDR_torch
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
def train(model, dataset, criterion, optimizer, mse, neg_sisdr, epoch):
    model.train()
    torch.set_grad_enabled(True)
    
    epoch_loss = 0
    epoch_mse_loss = 0
    epoch_sisdr = 0
    data_counter = 0
    
    for n_batch, train_batch in enumerate(dataset):
        optimizer.zero_grad()
        
        x, y = train_batch
        
        output = model(x)

        loss = criterion(output, y)
        
        print("LOSS = ", loss)
        
        epoch_mse_loss += mse(output, y).item()
        epoch_sisdr -= neg_sisdr(output, y).item()
        epoch_loss += loss.item()
        
        batch_size = x.shape[0]
        data_counter += batch_size

        loss.backward()
        
        if STORE_GRADIENT_NORM:
            with open(CKP_LOGS/f"train_epoch{epoch}.log", "a") as log:
                for layer in model.modules():
                    try:
                        name = layer.__str__()
                        mean_grad = np.mean(layer.weight.grad.detach().numpy())
                        print(">>> ",name, " grad =", mean_grad)
                        log.write(f"NAME : {name}\nLOSS : {loss.item()}\nGRADIENT VALUES MEAN: {mean_grad}\n\n")
                    except:
                        pass
        
        optimizer.step()
        
    epoch_loss /= data_counter
    epoch_mse_loss /= data_counter
    epoch_sisdr /= data_counter
    
    return epoch_loss, epoch_mse_loss, epoch_sisdr


def test(model, dataset, criterion, mse, neg_sisdr):
    with torch.no_grad():
        mean_loss = 0
        mean_mse_loss = 0
        epoch_sisdr = 0
        data_counter = 0
        
        for n_batch, test_batch in enumerate(dataset):
            x, y = test_batch
            
            output = model(x)

            loss = criterion(output, y)
            mean_mse_loss += mse(output, y).item()
            epoch_sisdr -= neg_sisdr(output, y).item()
            mean_loss += loss.item()
            
            batch_size = x.shape[0]
            data_counter += batch_size

            optimizer.step()
            
        mean_loss /= data_counter
        mean_mse_loss /= data_counter
        epoch_sisdr /= data_counter
    
    return mean_loss, mean_mse_loss, epoch_sisdr


def checkpoint(model, epoch, optimizer, lr_scheduler, best_loss, loss, ckp_dir, delta=1e-3):
    if loss <= best_loss + delta:
        torch.save(
            {
                'epoch': epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            },
            ckp_dir
        )


def fit(model, train_set, test_set, criterion, optimizer, lr_updater, epochs, history=None):
    
    if history is not None:
        # Train from checkpoint:
        
        train_loss_history = list(history["train_loss"].values)
        val_loss_history = list(history["val_loss"].values)
        train_mse_loss_history = list(history["train_mse_loss"].values)
        val_mse_loss_history = list(history["val_mse_loss"].values)
        train_sisdr_history = list(history["train_sisdr"].values)
        val_sisdr_history = list(history["val_sisdr"].values)
        lr_history = list(history["lr_history"].values)
        
        start_epoch = len(train_loss_history)
        print(f"\n>>> Restore training from EPOCH {start_epoch}\n")
    
    else:
        # Train from scratch:
        train_loss_history = list()
        val_loss_history = list()
        train_mse_loss_history = list()
        val_mse_loss_history = list()
        train_sisdr_history = list()
        val_sisdr_history = list()
        lr_history = list()
        
        start_epoch = 1
        print("\n>>> Begin training from scratch\n")
    
    mse = PITLossWrapper(pairwise_mse, pit_from="pw_mtx")
    neg_sisdr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    
    for epoch in range(start_epoch, start_epoch + epochs + 1):
        print(">>> EPOCH", epoch)
        
        train_loss, train_mse_loss, train_sisdr = train(model, train_set, criterion, optimizer, mse, neg_sisdr, epoch)
        lr_updater.step()
        
        val_loss, val_mse_loss, val_sisdr = test(model, test_set, criterion, mse, neg_sisdr)
        
        lr = lr_updater.get_last_lr()[0]
        
        train_loss_history.append(train_loss)
        train_mse_loss_history.append(train_mse_loss)
        val_loss_history.append(val_loss)
        val_mse_loss_history.append(val_mse_loss)
        train_sisdr_history.append(train_sisdr)
        val_sisdr_history.append(val_sisdr)
        lr_history.append(lr)
        
        best_loss = float('inf') if len(val_loss_history) == 0 else np.min(val_loss_history)
        
        # Save checkpoint:
        checkpoint(model, epoch, optimizer, lr_updater, best_loss, val_loss, CKP_PATH_MODEL)
        
        # Save weights every 10 epochs:
        if epoch % 10 == 0:
            ckp_dir = str(CKP_PATH/f"model_epoch{epoch}.pth")
            torch.save(
            {
                'epoch': epoch,
                'loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_updater.state_dict()
            },
            ckp_dir
        )

        # Store the learning curves
        history = pd.DataFrame(
                {
                    "train_loss": train_loss_history,
                    "val_loss": val_loss_history,
                    "train_mse_loss": train_mse_loss,
                    "val_mse_loss": val_mse_loss,
                    "train_sisdr": train_sisdr_history,
                    "val_sisdr": val_sisdr_history,
                    "lr": lr_history
                }
            )
        history.index.name = "epoch"
        history.to_csv(CKP_PATH_HISTORY)


################
### Train ######
################
fit(model, train_loader, test_loader, loss, optimizer, lr_updater, N_EPOCHS, history)