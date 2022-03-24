## Musdb18
To download musdb18 dataset, please go to the musdb folder and execute :  
wget https://zenodo.org/record/1117372/files/musdb18.zip?download=1 -O musdb18.zip

Then you'll need to unzip it with the command :  
unzip musdb18.zip

## Libraries

To install all of the dependencies, please use the command : 
pip -r install requirements.txt

## Command to run the code

python main.py -train=True -gradio=True

## Run the code on Artemis

To run the model on Artemis, please use script.sh :   
sbatch script.sh


## Structure of the files

• artemis.py : launch the training and testing of the VAE model.
• eval.py : implements functions to save and to listen to the sources computed by the model. Also can print the different metrics introduced in the report. 
• loss.py : contains the different loss function for the VAE model.
• main.py : implements a command line interface to easily run the model.
• model_enhance.py : contains the class of the Super Resolution model for audio enhancement.
• model_vae.py : contains the class of the VAE architecture.
• preprocess.py : contains all the function to transform the audio files into a usable dataset class.
• model_cluster.py : separates sources using ensemble clustering model.
• utils.py : plot the spectrogram of a signal


## Additional infos

Please note that we do not provide pretrained models because of their size (about 5Go.). So you will need to train and ensure that you have enough space on your disk.