
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import stempeg
from tqdm import tqdm
import numpy as np


def stereo_to_mono(audio):
    """transform a stereo signal to a mono signal

    Args:
        audio (array): audio array

    Returns:
        array: array in 1D
    """
    return (audio[:,0]+audio[:,1])/2


def stft(audio_data,n_freq=256, frame=128):
    """
    Compute the short time fourier transform of a signal
    Please note that audio_data needs to be a torch.Tensor
    Args:
        audio_data (torch.Tensor): _description_
        n_freq (int, optional): Number of frequency to keep. Defaults to 256.
        frame (int, optional): Number of frame to keep. Defaults to 128.

    Returns:
        tuple: returns the magnitude and the phase of the stft
    """
    stft_complex = torch.stft(audio_data, 2048, hop_length=512, win_length=512, window=torch.hann_window(512), center=True, pad_mode='reflect', normalized=True, onesided=None, return_complex=True)[:n_freq,...]
    mag, phase = stft_complex.abs(), stft_complex.angle()       
       
    return mag, phase

def slicing(slices_info, nb_frame, file_id):
    """Decompose a file into index to have audio of the same shape overlapping

    Args:
        slices_info (list): list
        nb_frame (int): number of frame in a slice
        file_id (int): index of the file

    Returns:
        list: list of index of the start of each slice
    """
    size_frame=128       
    n_max = 100
    mid = size_frame//2
    cpt = 0
    while mid + size_frame//2 < nb_frame and cpt < n_max:
        cpt += 1
        begin = mid - size_frame//2
        end = mid + size_frame//2 
        slices_info.append([file_id,begin,end])
        mid += size_frame//4
    return slices_info

class MusicDataset(Dataset):
    
    def __init__(self, filelist):
        
        self.slices_info = []
        self.filelist=filelist
        self.audio_data = {}

        for idx in tqdm(range(len(self.filelist))):

            S, rate = stempeg.read_stems(self.filelist[idx])

            mag,_ = stft(torch.tensor(stereo_to_mono(S[0])))
            source=[]
            for i in range(4):

              mag_source,_ = stft(torch.tensor(stereo_to_mono(S[i+1])))
              source.append(mag_source)
            
            source=np.stack(source)
            nb_frame = mag.shape[1]

            self.slices_info = slicing(self.slices_info, nb_frame, idx)
            self.audio_data[idx] = {'mix': mag.float(), 'sources': torch.from_numpy(source).float()}
          
    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
       
        mag, source = self.audio_data[self.slices_info[idx][0]]['mix'], self.audio_data[self.slices_info[idx][0]]['sources']
        mag_slice = mag[:, self.slices_info[idx][1]:self.slices_info[idx][2]]
        source_slice = source[..., self.slices_info[idx][1]:self.slices_info[idx][2]]
        
        return {'mix':mag_slice, 'sources':source_slice}
