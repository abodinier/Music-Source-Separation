import numpy as np
import gradio as gr
import nussl

def separate_audio_cluster(audio):
	"""separate sources using cluster algorithm

	Args:
		audio (path): input file

	Returns:
		tuple: (rate of audio, array with the audio)
	"""
	sr, data = audio
  
	audio_signal = nussl.AudioSignal(audio_data_array=data.astype(np.int16) )
  
	separators = [
	  nussl.separation.primitive.FT2D(audio_signal),
	  nussl.separation.primitive.HPSS(audio_signal)
	  # ,
	  # nussl.separation.primitive.Melodia(audio_signal),
  ]

	weights = [2, 1]
	returns = [[1], [0]]

	fixed_centers = np.array([
		[0 for i in range(sum(weights))],
		[1 for i in range(sum(weights))],
	])
	ensemble = nussl.separation.composite.EnsembleClustering(
		audio_signal, 2, separators=separators, init=fixed_centers,
		fit_clusterer=False, weights=weights, returns=returns)
	ensemble.clusterer.cluster_centers_ = fixed_centers
	estimates = ensemble()

	estimates = {
		f'Cluster {i}': e for i, e in enumerate(estimates)
	}
	n = estimates['Cluster 0'].audio_data[0].shape[0]

	print("Number of sources separated: ",len(estimates))
	return (sr,(estimates['Cluster 0'].audio_data[0]* 32767).astype(np.int16)),(sr,(estimates['Cluster 1'].audio_data[0]* 32767).astype(np.int16)), (sr,estimates['Cluster 1'].audio_data[0].astype(np.int16)), (sr,estimates['Cluster 1'].audio_data[0].astype(np.int16))
