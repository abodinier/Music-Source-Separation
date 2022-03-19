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
from torch.nn.functional import l1_loss, mse_loss

from asteroid.data import MUSDB18Dataset

from kaituoxu.conv_tasnet import ConvTasNet

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
SAMPLES_PER_TRACK = CFG["samples_per_track"]
RANDOM_SEGMENT = CFG["random_segment"]
RANDOM_TRACK_MIX = CFG["random_track_mix"]
TARGETS = CFG["targets"]
N_SRC = len(TARGETS)
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
print("\t>>> CKP name = ", CKP_PATH.name)
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
    samples_per_track=SAMPLES_PER_TRACK,
    random_segments=RANDOM_SEGMENT,
    random_track_mix=RANDOM_TRACK_MIX,
    sample_rate=SAMPLE_RATE,
    size=SIZE
)
TRAIN_LOADER = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
print(">>> Training Dataloader ready\n")

test_dataset = MUSDB18Dataset(
    root=DATA_DIR.__str__(),
    targets=TARGETS,
    suffix=".mp4",
    split="test",
    subset=None,
    segment=SEGMENT_SIZE,
    samples_per_track=SAMPLES_PER_TRACK,
    random_segments=RANDOM_SEGMENT,
    random_track_mix=RANDOM_TRACK_MIX,
    sample_rate=SAMPLE_RATE,
    size=SIZE
)
TEST_LOADER = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
print(">>> TEST Dataloader ready\n")


################
##### MODEL ####
################
MODEL = ConvTasNet(
    C=N_SRC,
    X=X,
    R=R,
    B=B,
    H=H,
    P=P,
    L=L,
    N=N,
    stride=STRIDE,
    mask_nonlinear="softmax",
    device=DEVICE
).to(DEVICE)

LOSS = eval(CFG["loss"])
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LR)
LR_UPDATER = lr_scheduler.ReduceLROnPlateau(OPTIMIZER, patience=5, factor=0.5)
HISTORY = None


if args.restore is not None:
    MODEL.load_state_dict(torch.load(CKP_PATH_MODEL)["model_state_dict"])
    OPTIMIZER.load_state_dict(torch.load(CKP_PATH_MODEL)["optimizer_state_dict"])
    LR_UPDATER.load_state_dict(torch.load(CKP_PATH_MODEL)["lr_scheduler"])
    HISTORY = pd.read_csv(CKP_PATH_HISTORY)


################
### TRAINING ###
################
def train(model, dataset, criterion, optimizer, epoch):
    use_cuda = CFG["device"] == "cuda"
    model.train()
    
    epoch_loss = 0
    data_counter = 0
    
    if use_cuda:
        scaler = torch.cuda.amp.GradScaler()
    
    for x, y in dataset:
        batch_size = x.shape[0]
        
        save_data_example(x, y, epoch)
    
        optimizer.zero_grad()
        if use_cuda:
            x = x.to(DEVICE, dtype=torch.float16)
            y = y.to(DEVICE, dtype=torch.float16)
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y, reduction='none').sum(axis=(0, 1, 2))  # sum over batches, sources and frames
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            loss = criterion(output, y, reduction='none').sum(axis=(0, 1, 2))  # sum over batches, sources and frames
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        data_counter += batch_size
        
        if STORE_GRADIENT_NORM:
            save_gradient_norms(model, loss, epoch)
        
        if SAVE_WEIGHTS_EACH_EPOCH:
            save_ckp(model, optimizer, loss, epoch)
        
    epoch_loss /= data_counter
    
    return epoch_loss


def test(model, dataset, criterion):
    use_cuda = CFG["device"] == "cuda"
    model.eval()
    
    with torch.no_grad():
        mean_loss = 0
        data_counter = 0
        
        for x, y in dataset:
            batch_size = x.shape[0]
            
            if use_cuda:
                x = x.to(DEVICE, dtype=torch.float16)
                y = y.to(DEVICE, dtype=torch.float16)
                with torch.cuda.amp.autocast():
                    output = model(x)
                    loss = criterion(output, y).sum(axis=(0, 1))
            else:
                output = model(x)
                loss = criterion(output, y).sum(axis=(0, 1))
            
            mean_loss += loss.item()
            data_counter += batch_size
        
        mean_loss /= data_counter
    
    return mean_loss


def fit(model, train_set, test_set, criterion, optimizer, lr_updater, epochs, history=None):
    
    if history is not None:
        # Train from checkpoint:
        
        train_loss_history = list(history["train_loss"].values)
        val_loss_history = list(history["val_loss"].values)
        lr_history = list(history["lr_history"].values)
        
        start_epoch = len(train_loss_history)
        print(f"\n>>> Restore training from EPOCH {start_epoch}\n")
    
    else:
        # Train from scratch:
        train_loss_history = list()
        val_loss_history = list()
        lr_history = list()
        
        start_epoch = 1
        print("\n>>> Begin training from scratch\n")
    
    for epoch in range(start_epoch, start_epoch + epochs + 1):
        print(">>> EPOCH", epoch)
        
        lr = optimizer.param_groups[0]['lr']
        
        train_loss = train(model, train_set, criterion, optimizer, epoch)
        val_loss = test(model, test_set, criterion)
        
        lr_updater.step(val_loss)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        lr_history.append(lr)
        
        
        # Save checkpoint:
        best_loss = float('inf') if len(val_loss_history) == 0 else np.min(val_loss_history)
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
                    "lr": lr_history
                }
            )
        history.index.name = "epoch"
        history.to_csv(CKP_PATH_HISTORY)


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
                name = path/f"{TARGETS[j]}.wav"
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
fit(
    model=MODEL,
    train_set=TRAIN_LOADER,
    test_set=TEST_LOADER,
    criterion=LOSS,
    optimizer=OPTIMIZER,
    lr_updater=LR_UPDATER,
    epochs=N_EPOCHS,
    history=HISTORY
)