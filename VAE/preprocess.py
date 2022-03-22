
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import stempeg
from tqdm import tqdm
import numpy as np


def stereo_to_mono(audio):

  return (audio[:,0]+audio[:,1])/2


def stft(audio_data,n_freq=256, frame=128):

    stft_complex = torch.stft(audio_data, 2048, hop_length=512, win_length=512, window=torch.hann_window(512), center=True, pad_mode='reflect', normalized=True, onesided=None, return_complex=True)[:n_freq,...]
    mag, phase = stft_complex.abs(), stft_complex.angle()       
       
    return mag, phase

def slicing(slices_info, nb_frame, file_id):
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
