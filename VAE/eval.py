from preprocess import *
import IPython
from loss import *
import museval

def eval_audio(idx,filelist, n_freq=256, length=1000000):
  """return a dictionnary with the different sources of a voice, and with their stft.
  Dictionnary follows the same syntax as MUSDB18 dataset i.e. mix, drums, bass, other and vocals sources.
  Args:
      idx (int): index of the file in the list to return the dictionnary
      filelist (list): Contains all the path of the file
      n_freq (int, optional): number of frequency kept in the stft. Defaults to 256.
      length (int, optional): size of the audio. Defaults to 1000000.

  Returns:
      track, track_stft: dict
  """
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
  """
  Returns the different sources from the vae model
  
  Args:
      model : vae model
      track (dict): contains the stft of the song
      device (str, optional):  Defaults to 'cuda'.

  Returns:
     s_vaem, x_vaem, s_vae, x_vae : masked sources, reconstruced masked song, sources, reconstructed song
  """
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
  """Same function as separate song but recrete the full songs.
  Use this function with gradio.
  Args:
      model : vae model
      track (dict): dictionnary of stft
      device (str, optional): Defaults to 'cpu'.
      n_freq (int, optional): Defaults to 256.
      timeframe (int, optional): Defaults to 128.
      rate (int, optional):  Defaults to 20000.

  Returns:
       Mix, S1, S2, S3, S4: Reconstructed song and the 4 sources.
  """
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

def eval(model,file_list):
  """Compute the SDR, ISR, SIR, SAR, and PERM metric on the files from file_list, using the model.

  Args:
      model : source separation model
      file_list (list): contains the path of all the songs.
  """
  j = torch.complex(torch.tensor(0.),torch.tensor(1.))

  SDR = 0
  ISR = 0
  SIR = 0
  SAR = 0
  PERM = 0
  n = len(file_list)

  for i in range(n):
    track, track_stft = eval_audio(i, file_list)
    s_vaem, x_vaem, s_vae, x_vae = separate_source(model, track_stft)
    s = s_vaem[0][0].to('cpu') * ( torch.cos(track_stft['mix'][1][:256,:128]) + j* torch.sin(track_stft['mix'][1][:256,:128]))
    estimates = torch.istft(s.detach(),511,normalized=True)
    sdr,isr,sir,sar,perm=museval.metrics.bss_eval(track['mix'][:estimates.shape[0]].numpy()+0.000001,estimates.numpy()+0.00001)
    SDR +=sdr[0][0] /n
    ISR +=isr[0][0] /n
    SIR +=sir[0][0] /n
    SAR +=sar[0][0] /n
    PERM +=perm[0][0] /n

  print(SDR)
  print(ISR)
  print(SIR)
  print(SAR)
  print(PERM)

  return SDR,ISR,SIR,SAR,PERM