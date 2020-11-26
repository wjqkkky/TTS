#!/usr/bin/env python
# coding: utf-8

# This is a noteboook used to generate the speaker embeddings with the CorentinJ GE2E model trained with Angular Prototypical loss for multi-speaker training.
# 
# Before running this script please DON'T FORGET:
# - to set the right paths in the cell below.
# 
# Repositories:
# - TTS: https://github.com/mozilla/TTS
# - CorentinJ GE2E: https://github.com/Edresson/GE2E-Speaker-Encoder

# In[2]:


import os
import importlib
import random
import librosa
import torch

import numpy as np
from TTS.utils.io import load_config
from tqdm import tqdm
from TTS.tts.tts_utils.speakers import save_speaker_mapping, load_speaker_mapping

# you may need to change this depending on your system
os.environ['CUDA_VISIBLE_DEVICES']='0'


# In[3]:


# Clone encoder 
#os.system('git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git')
#os.chdir('Real-Time-Voice-Cloning/')


# In[4]:


#Install voxceleb_trainer Requeriments
#os.system('pip install umap-learn visdom webrtcvad librosa>=0.5.1 matplotlib>=2.0.2 numpy>=1.14.0  scipy>=1.0.0  tqdm sounddevice Unidecode inflect multiprocess numba')


# In[5]:


#Download encoder Checkpoint
#os.system('wget https://github.com/Edresson/Real-Time-Voice-Cloning/releases/download/checkpoints/pretrained.zip')
#os.system('unzip pretrained.zip')


# In[6]:

import sys 
sys.path.insert(1, 'Real-Time-Voice-Cloning/')
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from pathlib import Path


# In[7]:


print("Preparing the encoder, the synthesizer and the vocoder...")
encoder.load_model(Path('Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt'))
print("Testing your configuration with small inputs.")
# Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
# sampling rate, which may differ.
# If you're unfamiliar with digital audio, know that it is encoded as an array of floats 
# (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
# The sampling rate is the number of values (samples) recorded per second, it is set to
# 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond 
# to an audio of 1 second.
print("\tTesting the encoder...")

wav = np.zeros(encoder.sampling_rate)    
embed = encoder.embed_utterance(wav)
print(embed.shape)

# Embeddings are L2-normalized (this isn't important here, but if you want to make your own 
# embeddings it will be).
#embed /= np.linalg.norm(embed) # for random embedding


# In[15]:


SAVE_PATH = '../'


# In[12]:


# Set constants
#DATASETS_NAME = ['vctk','libri_tts', 'libri_tts', 'libri_tts', ] # list the datasets
#DATASETS_PATH = ['../../../../datasets/VCTK-Corpus-removed-silence/', '../../../../datasets/LibriTTS/LibriTTS/dataset/', '../../../../datasets/LibriTTS/test/LibriTTS/test-other/', '../../../../datasets/LibriTTS/test/LibriTTS/test-clean/']
#DATASETS_METAFILE = ['', None, None, None]
DATASETS_NAME = ['vctk', 'brspeech', 'brspeech'] # list the datasets
DATASETS_PATH = ['../../../../datasets/VCTK-Corpus-removed-silence/', '../../../../datasets/BRSpeech-3-Speakers-Paper/TTS-Portuguese_Corpus/', '../../../../datasets/BRSpeech-3-Speakers-Paper/TTS-Portuguese_Corpus/']

DATASETS_METAFILE = ['', 'train_TTS-Portuguese_Corpus_metadata.csv','test_TTS-Portuguese_Corpus_metadata.csv']

USE_CUDA = True


# In[18]:


#Preprocess dataset
meta_data = []
for i in range(len(DATASETS_NAME)):
    preprocessor = importlib.import_module('TTS.tts.datasets.preprocess')
    preprocessor = getattr(preprocessor, DATASETS_NAME[i].lower())
    meta_data += preprocessor(DATASETS_PATH[i],DATASETS_METAFILE[i])
      
meta_data= list(meta_data)

meta_data = meta_data
embeddings_dict = {}
len_meta_data= len(meta_data)
for i in tqdm(range(len_meta_data)):
    _, wave_file_path, speaker_id = meta_data[i]
    wav_file_name = os.path.basename(wave_file_path)
    # Extract Embedding
    preprocessed_wav = encoder.preprocess_wav(wave_file_path)
    file_embedding = encoder.embed_utterance(preprocessed_wav)
    embeddings_dict[wav_file_name] = [file_embedding.reshape(-1).tolist(), speaker_id]
    del file_embedding


# In[19]:


# create and export speakers.json  and aplly a L2_norm in embedding
speaker_mapping = {sample: {'name': embeddings_dict[sample][1], 'embedding':embeddings_dict[sample][0]} for i, sample in enumerate(embeddings_dict.keys())}
save_speaker_mapping(SAVE_PATH, speaker_mapping)


# In[20]:


#test load integrity
speaker_mapping_load = load_speaker_mapping(SAVE_PATH)
assert speaker_mapping == speaker_mapping_load
print("The file speakers.json has been exported to ",SAVE_PATH, ' with ', len(embeddings_dict.keys()), ' samples')


# In[ ]:




