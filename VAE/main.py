import argparse
from model_cluster import *
import artemis
import gradio as gr



def main(args):
	print("Starting program.")

	if args.train:
		print("Launching the training.")
		artemis.launch_model(save=args.save, plot=args.plot, load=args.load)

	
	if args.gradio:
		print('Opening gradio interface.')
		inp = gr.inputs.Audio( source="upload", type="numpy", label=None, optional=False)
		out1 = gr.outputs.Audio(type="auto", label='Vocals')
		out2 = gr.outputs.Audio(type="auto", label='Drums')
		out3 = gr.outputs.Audio(type="auto", label='Bass')
		out4 = gr.outputs.Audio(type="auto", label='Other')
		iface = gr.Interface(separate_audio_cluster,inp, [out1,out2,out3,out4],description="Please upload a song with several sources. Then click on submit to separate the sources.",  title="Source separation",theme='grass')
		iface.launch(debug=False)

	print("Program finished sucessfully.")









if __name__=="__main__":


	parser = argparse.ArgumentParser()

	parser.add_argument("-t", "--train", metavar="T", default=True, type=bool,
		help="Set to True to train the Variational Autoencoder on musdb18 data.")

	parser.add_argument("-s", "--save", metavar="S", default=None, type=str,
		help="Set to True to save the model.")

	parser.add_argument("-l", "--load", metavar="L", default=None, type=str,
		help="Path of the model trained to load.")


	parser.add_argument("-p", "--plot", default=None, type=str, 
		help="If set to True will plot loss function.")

	parser.add_argument("-g", "--gradio", metavar="G", default=False, type=bool,
		help="If set to True will launch a gradio web interface to play with the model.")

	args = parser.parse_args()
	main(args)
