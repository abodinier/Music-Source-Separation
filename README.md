# IA321 - Artificial Intelligence Project
## Topic: Blind Source Separation applied to music

* Authors: Grégoire Hugon, Davy Simeu, Thibaut Vaillant, Thomas Nigoghossian, Alexandre Bodinier
* Supervisor: Franchi Gianni (Prof. U2IS, ENSTA Paris, Institut Polytechnique de Paris)

**Blind Source Separation** for music is the task of **isolating the multiple components of a song**: vocals, bass, drums and any other accompaniments. Typically, all the instruments and voices are recorded individually in a studio during the recording session and then arranged together during the mixing session.

## Useful music source separation repositories:
1. Asteroid: https://github.com/asteroid-team/asteroid
2. **SigSep** Open-Unmix: https://github.com/sigsep/open-unmix-pytorch
3. **Facebook** Demucs: https://github.com/facebookresearch/demucs 
4. **Deezer** Spleeter: https://github.com/deezer/spleeter

## Requirements:
In order to read STEM files, you must install ffmpeg.
If you use conda, run `conda install -c conda-forge ffmpeg`, on MacOs, run `brew install ffmpeg`. Otherwise, please refer to [the offical repository](https://github.com/faroit/stempeg) for more information.
