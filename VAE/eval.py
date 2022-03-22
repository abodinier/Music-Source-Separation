from preprocess import *
import IPython
from loss import *

def eval_audio(idx,filelist, n_freq=256, length=1000000):

  audio_data , rate = stempeg.read_stems(filelist[idx],dtype=np.float32)
  audio_mix = stereo_to_mono(audio_data[0][:length])
  audio_drums = stereo_to_mono(audio_data[1][:length])
  audio_bass = stereo_to_mono(audio_data[2][:length])
  audio_other = stereo_to_mono(audio_data[3][:length])
  audio_vocals = stereo_to_mono(audio_data[4][:length])


  track = dict()
  track_stft = dict()

  track['mix'] = torch.tensor(audio_mix)
  track['drums'] = torch.tensor(audio_drums)
  track['bass'] = torch.tensor(audio_bass)
  track['other'] =torch.tensor(audio_other)
  track['vocals'] = torch.tensor(audio_vocals)

  track_stft['mix'] = stft(torch.tensor(audio_mix), n_freq=n_freq)
  track_stft['drums'] = stft(torch.tensor(audio_drums), n_freq=n_freq)
  track_stft['bass'] = stft(torch.tensor(audio_bass), n_freq=n_freq)
  track_stft['other'] = stft(torch.tensor(audio_other), n_freq=n_freq)
  track_stft['vocals'] = stft(torch.tensor(audio_vocals), n_freq=n_freq)

  print("Mix")
  IPython.display.display(IPython.display.Audio(track['mix'], rate=rate))
  print("drums")
  IPython.display.display(IPython.display.Audio(track['drums'], rate=rate))
  print("bass")
  IPython.display.display(IPython.display.Audio(track['bass'], rate=rate))
  print("other")
  IPython.display.display(IPython.display.Audio(track['other'], rate=rate))
  print("vocals")
  IPython.display.display(IPython.display.Audio(track['vocals'], rate=rate))

  return track, track_stft



def separate_source(model, track, device='cuda'):
  model.eval()
  s1 =  torch.tensor(track['drums'][0][:256,:128])
  s2 =  torch.tensor(track['vocals'][0][:256,:128])
  s3 =  torch.tensor(track['bass'][0][:256,:128])
  s4 =  torch.tensor(track['other'][0][:256,:128])

  s_tru = torch.stack([s1,s2,s3,s4]).to(device)
  s_tru = s_tru[None,...]
  model.eval()
  x_vae, mu_z, logvar_z, s_vae = model(track['mix'][0][:256,:128].to(device))
  x_vae = x_vae.view(-1,1,256,128)
  s_vae = s_vae.view(-1,4,256,128)

  s_vae = optimal_permute(s_vae,s_tru)

  # Create masks
  s_vaem = vae_masks(s_vae,track['mix'][0][:256,:128].to(device))
  x_vaem = s_vaem.sum(1).unsqueeze(1)

  #Reconstruct
  j = torch.complex(torch.tensor(0.),torch.tensor(1.))
  rate = 18000

  mix = x_vaem[0][0].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  mix = torch.istft(mix.detach(),511,normalized=True)
  
  s1 = s_vaem[0][0].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  s1 = torch.istft(s1.detach(),511,normalized=True)
  s2 = s_vaem[0][1].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  s2 = torch.istft(s2.detach(),511,normalized=True)
  s3 = s_vaem[0][2].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  s3 = torch.istft(s3.detach(),511,normalized=True)
  s4 = s_vaem[0][3].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  s4 = torch.istft(s4.detach(),511,normalized=True)


  print("Mix")
  IPython.display.display(IPython.display.Audio(mix, rate=rate))
  print("drums")
  IPython.display.display(IPython.display.Audio(s1, rate=rate))
  print("bass")
  IPython.display.display(IPython.display.Audio(s2, rate=rate))
  print("other")
  IPython.display.display(IPython.display.Audio(s3, rate=rate))
  print("vocals")
  IPython.display.display(IPython.display.Audio(s4, rate=rate))

  return s_vaem, x_vaem, s_vae, x_vae


def separate_full_song(model,track, device='cpu', n_freq=256, timeframe=128, rate=20000):

  len_song = track['drums'][0].shape[1]
  nbr_splits = len_song // timeframe 
  Mix = []
  S1 = []
  S2 = []
  S3 = []
  S4 = []

  for i in range(nbr_splits):
    s1 = track['drums'][0][:,i*timeframe:(i+1) * timeframe]
    s2 = track['vocals'][0][:,i*timeframe:(i+1)*timeframe]
    s3 = torch.zeros((n_freq,timeframe))
    s4 = torch.zeros((n_freq,timeframe))
    data = s1 + s2
    data_phase = track['drums'][1][:,i*timeframe:(i+1)*timeframe] +  track['vocals'][1][:,i*timeframe:(i+1)*timeframe] 
    s_tru = torch.stack([s1,s2,s3,s4]).to(device)
    s_tru = s_tru[None,...]
    model.eval()
    x_vae, mu_z, logvar_z, s_vae = model(data.float().to(device))
    x_vae = x_vae.view(-1,1,n_freq,timeframe)
    s_vae = s_vae.view(-1,4,n_freq,timeframe)

    s_vae = optimal_permute(s_vae,s_tru)

    # Create masks
    s_vaem = vae_masks(s_vae,data.to(device))
    x_vaem = s_vaem.sum(1).unsqueeze(1)

    #Reconstruct
    j = torch.complex(torch.tensor(0.),torch.tensor(1.))

    mix = x_vaem[0][0].to('cpu') * ( torch.cos(data_phase) + j* torch.sin(data_phase))
    mix = torch.istft(mix.detach(),511,normalized=True)
    j = torch.complex(torch.tensor(0.),torch.tensor(1.))

    mix = x_vaem[0][0].to('cpu') * ( torch.cos(data_phase) + j* torch.sin(data_phase))
    mix = torch.istft(mix.detach(),511,normalized=True)

    s1 = s_vaem[0][0].to('cpu') * ( torch.cos(data_phase) + j* torch.sin(data_phase))
    s1 = torch.istft(s1.detach(),511,normalized=True)
    s2 = s_vaem[0][1].to('cpu') * ( torch.cos(data_phase) + j* torch.sin(data_phase))
    s2 = torch.istft(s2.detach(),511,normalized=True)
    s3 = s_vaem[0][2].to('cpu') * ( torch.cos(data_phase) + j* torch.sin(data_phase))
    s3 = torch.istft(s3.detach(),511,normalized=True)
    s4 = s_vaem[0][3].to('cpu') * ( torch.cos(data_phase) + j* torch.sin(data_phase))
    s4 = torch.istft(s4.detach(),511,normalized=True)

    Mix += list(mix.numpy())
    S1 += list(s1.numpy())
    S2 += list(s2.numpy())
    S3 += list(s3.numpy())
    S4 += list(s4.numpy())


  
  print("Mix")
  IPython.display.display(IPython.display.Audio(np.array(Mix), rate=rate))
  print("drums")
  IPython.display.display(IPython.display.Audio(np.array(S1), rate=rate))
  print("bass")
  IPython.display.display(IPython.display.Audio(np.array(S2), rate=rate))
  print("other")
  IPython.display.display(IPython.display.Audio(np.array(S3), rate=rate))
  print("vocals")
  IPython.display.display(IPython.display.Audio(np.array(S4), rate=rate))

  return Mix, S1, S2, S3, S4
