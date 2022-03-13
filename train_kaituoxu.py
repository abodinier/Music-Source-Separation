import time
import yaml
import torch
import shutil
import random
import argparse
import numpy as np
import pandas as pd
from torch import optim
from pathlib import Path
from scipy.io import wavfile
from datetime import datetime
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from kaituoxu.conv_tasnet import ConvTasNet
from kaituoxu.pit_criterion import cal_loss as si_snr
from torch.nn.functional import l1_loss, mse_loss

from asteroid.data import MUSDB18Dataset
from asteroid.losses import PITLossWrapper
from asteroid.losses import pairwise_mse


#####################
##### ARGS ##########
#####################
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_path", default="cfg.yaml", type=str)
parser.add_argument("--data_dir", default="musdb_data", type=str)
parser.add_argument("--ckpdir", default="weights", type=str)
parser.add_argument("--restore", default=None, type=str)
parser.add_argument("--description", default=None, type=str)
args = parser.parse_args()

if args.restore is not None:
    CKP_PATH = Path(args.restore)
else:
    time.sleep(1. + 20 * random.random())
    CKP_PATH = Path(args.ckpdir)/f"training_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')[:-3]}"
    while CKP_PATH.is_dir(): # avoid problems creating multiple training dir simultaneously
        time.sleep(1. + 10 * random.random())
        CKP_PATH = Path(args.ckpdir)/f"training_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')[:-3]}"

CKP_PATH.mkdir(parents=True)

CKP_LOGS = CKP_PATH/"logs"
CKP_PATH_MODEL = CKP_PATH/"model.pth"
CKP_PATH_HISTORY = CKP_PATH/"history.csv"
CKP_PATH_CFG = CKP_PATH/f"{Path(args.cfg_path).name}"

if not CKP_LOGS.is_dir():
    CKP_LOGS.mkdir(parents=True)

if not CKP_PATH_CFG.exists():
    shutil.copy(args.cfg_path, CKP_PATH_CFG)

with open(str(CKP_PATH_CFG), 'r') as file:
    CFG = yaml.load(file, Loader=yaml.FullLoader)

if args.description is not None:
    with open(str(CKP_PATH/"description"), "w") as desc:
        desc.write(args.description)

DATA_DIR = Path(args.data_dir)
SEGMENT_SIZE = CFG["segment_size"]
RANDOM_TRACK_MIX = CFG["random_track_mix"]
TARGETS = CFG["targets"]
N_SRC = len(TARGETS)
LOSS = eval(CFG["loss"])
STORE_GRADIENT_NORM = CFG["store_gradient_norm"]
VERBOSE = CFG["verbose"]
SAVE_WEIGHTS_EACH_EPOCH = CFG["save_weights_each_epoch"]

#####################
##### HYPER-PARAMETERS
#####################
SAMPLE_RATE = CFG["sample_rate"]
SIZE = None if CFG["size"] == -1 else CFG["size"]
LR = CFG["learning_rate"]
N_EPOCHS = CFG["n_epochs"]
TRAIN_BATCH_SIZE = CFG["train_batch_size"]
TEST_BATCH_SIZE = CFG["test_batch_size"]
NUM_WORKERS = CFG["num_workers"]
if CFG["device"] == "cuda" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

X = CFG["X"]
R = CFG["R"]
B = CFG["B"]
H = CFG["H"]
Sc = CFG["Sc"]
P = CFG["P"]
L = CFG["L"]
N = CFG["N"]
STRIDE = CFG["stride"]
CLIP = CFG["gradient_clipping"]

print("\n>>> PARAMETERS")
for key, value in CFG.items():
    print(f"\t>>> {key.upper()} -> {value}")
print("\t>>> DEVICE = ", DEVICE)
print("\n\n")

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
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS)
print(">>> Training Dataloader ready\n")

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
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS)
print(">>> TEST Dataloader ready\n")


################
##### MODEL ####
################
model = ConvTasNet(
    C=N_SRC,
    X=X,
    R=R,
    B=B,
    H=H,
    P=P,
    L=L,
    N=N,
    stride=STRIDE,
    mask_nonlinear="softmax"
).to(DEVICE)

loss = LOSS
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_updater = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
history = None


if args.restore is not None:
    model.load_state_dict(torch.load(CKP_PATH_MODEL)["model_state_dict"])
    optimizer.load_state_dict(torch.load(CKP_PATH_MODEL)["optimizer_state_dict"])
    lr_updater.load_state_dict(torch.load(CKP_PATH_MODEL)["lr_scheduler"])
    history = pd.read_csv(CKP_PATH_HISTORY)


################
### TRAINING ###
################
def train(model, dataset, criterion, optimizer, mse, epoch):
    model.train()
    torch.set_grad_enabled(True)
    
    epoch_loss = 0
    epoch_mse_loss = 0
    epoch_snr = 0
    data_counter = 0
    
    for n_batch, train_batch in enumerate(dataset):
        optimizer.zero_grad()
        
        x, y = train_batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        save_data_example(x, y, epoch)
    
        batch_size = x.shape[0]
        length = x.shape[-1]
        signal_length = length * torch.ones(batch_size).to(DEVICE)
        
        output, loss, max_snr = forward(model, x, y, signal_length, criterion, DEVICE)
        loss.backward()
        
        epoch_mse_loss += mse(output, y).item()
        epoch_loss += loss.item()
        epoch_snr += max_snr.sum().item()
        
        data_counter += batch_size

        
        if STORE_GRADIENT_NORM:
            save_gradient_norms(model, loss, epoch)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        
        if SAVE_WEIGHTS_EACH_EPOCH:
            save_ckp(model, optimizer, loss, epoch)
        
    epoch_loss /= data_counter
    epoch_mse_loss /= data_counter
    epoch_snr /= data_counter
    
    return epoch_loss, epoch_mse_loss, epoch_snr


def test(model, dataset, criterion, mse):
    model.eval()
    
    with torch.no_grad():
        mean_loss = 0
        mean_mse_loss = 0
        mean_snr = 0
        data_counter = 0
        
        for n_batch, test_batch in enumerate(dataset):
            x, y = test_batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            batch_size = x.shape[0]
            length = x.shape[-1]
            signal_length = length * torch.ones(batch_size)
            
            output, loss, max_snr = forward(model, x, y, signal_length, criterion, DEVICE)
            
            mean_mse_loss += mse(output, y).item()
            mean_loss += loss.item()
            mean_snr += max_snr.sum().item()
            
            data_counter += batch_size

            optimizer.step()
            
        mean_loss /= data_counter
        mean_mse_loss /= data_counter
        mean_snr /= data_counter
    
    return mean_loss, mean_mse_loss, mean_snr


def fit(model, train_set, test_set, criterion, optimizer, lr_updater, epochs, history=None):
    
    if history is not None:
        # Train from checkpoint:
        
        train_loss_history = list(history["train_loss"].values)
        val_loss_history = list(history["val_loss"].values)
        train_mse_loss_history = list(history["train_mse_loss"].values)
        val_mse_loss_history = list(history["val_mse_loss"].values)
        train_snr_history = list(history["train_snr"].values)
        val_snr_history = list(history["val_snr"].values)
        lr_history = list(history["lr_history"].values)
        
        start_epoch = len(train_loss_history)
        print(f"\n>>> Restore training from EPOCH {start_epoch}\n")
    
    else:
        # Train from scratch:
        train_loss_history = list()
        val_loss_history = list()
        train_mse_loss_history = list()
        val_mse_loss_history = list()
        train_snr_history = list()
        val_snr_history = list()
        lr_history = list()
        
        start_epoch = 1
        print("\n>>> Begin training from scratch\n")
    
    mse = PITLossWrapper(pairwise_mse, pit_from="pw_mtx")
    
    for epoch in range(start_epoch, start_epoch + epochs + 1):
        print(">>> EPOCH", epoch)
        
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_mse_loss, train_snr = train(model, train_set, criterion, optimizer, mse, epoch)
        
        val_loss, val_mse_loss, val_snr = test(model, test_set, criterion, mse)
        lr_updater.step(val_loss)
        
        train_loss_history.append(train_loss)
        train_mse_loss_history.append(train_mse_loss)
        val_loss_history.append(val_loss)
        val_mse_loss_history.append(val_mse_loss)
        train_snr_history.append(train_snr)
        val_snr_history.append(val_snr)
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
                    "train_snr": train_snr_history,
                    "val_snr": val_snr_history,
                    "lr": lr_history
                }
            )
        history.index.name = "epoch"
        history.to_csv(CKP_PATH_HISTORY)

def forward(model, x, y, signal_length, criterion, device):
    if CFG["device"] == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(x)

            if CFG["loss"] == "si_snr":
                loss, max_snr, estimate_source, reorder_estimate_source = criterion(output, y, signal_length)
            if CFG["loss"] in ("l1_loss", "mse_loss"):
                loss = criterion(output, y)
                max_snr = torch.zeros(2)
    else:
        output = model(x)

        if CFG["loss"] == "si_snr":
            loss, max_snr, estimate_source, reorder_estimate_source = criterion(output, y, signal_length)
        if CFG["loss"] in ("l1_loss", "mse_loss"):
            loss = criterion(output, y)
            max_snr = torch.zeros(2)
    
    return output, loss, max_snr

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

def save_data_example(x, y, epoch):
    if epoch == 1:
        for i in range(x.shape[0]):
            path = CKP_PATH/f"epoch_{epoch}_track_{i}"
            if not path.is_dir():
                path.mkdir()
            wavfile.write(path/"mixture.wav", SAMPLE_RATE, x[i].cpu().detach().numpy())
            for j, s in enumerate(y[i]):
                name = path/f"{j}.wav"
                wavfile.write(str(name), SAMPLE_RATE, s.cpu().detach().numpy())

def save_gradient_norms(model, loss, epoch):
    with open(CKP_LOGS/f"train_epoch{epoch}.log", "a") as log:
        for layer in model.modules():
            try:
                name = layer.__str__()
                min_grad = np.min(np.abs(layer.weight.grad.cpu().detach().numpy()))
                mean_grad = np.mean(np.abs(layer.weight.grad.cpu().detach().numpy()))
                max_grad = np.max(np.abs(layer.weight.grad.cpu().detach().numpy()))
                info = f">>> NAME : {name} | LOSS = {loss.item()} | min grad = {min_grad} | max grad = {max_grad} | mean grad = {mean_grad}\n"
                if VERBOSE == 1:
                    print(info)
                log.write(info)
            except Exception as e:
                pass

def save_ckp(model, optimizer, loss, epoch):
    torch.save(
        {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        CKP_PATH/f"model_{epoch}.pth"
    )

################
### Train ######
################
fit(model, train_loader, test_loader, loss, optimizer, lr_updater, N_EPOCHS, history)