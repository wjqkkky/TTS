#!/usr/bin/env python
# coding: utf-8

# This is a noteboook used to generate the speaker embeddings with the AngleProto speaker encoder model for multi-speaker training.
# 
# Before running this script please DON'T FORGET: 
# - to set file paths.
# - to download related model files from TTS.
# - download or clone related repos, linked below.
# - setup the repositories. ```python setup.py install```
# - to checkout right commit versions (given next to the model) of TTS.
# - to set the right paths in the cell below.
# 
# Repository:
# - TTS: https://github.com/mozilla/TTS

# In[ ]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import os
import importlib
import random
import librosa
import torch

import numpy as np
from tqdm import tqdm
from TTS.tts.utils.speakers import save_speaker_mapping, load_speaker_mapping
from TTS.speaker_encoder.model import SpeakerEncoder
# you may need to change this depending on your system
os.environ['CUDA_VISIBLE_DEVICES']='0'

from glob import glob

from TTS.tts.utils.speakers import save_speaker_mapping, load_speaker_mapping
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config


# You should also adjust all the path constants to point at the relevant locations for you locally

# In[ ]:


MODEL_RUN_PATH = "../../../../datasets/training/speaker_encoder/LibriTTS-common-voice-voxceleb_angle_proto/"
JSON_OUTPUT = "../../../../datasets/training/speaker_encoder/LibriTTS-common-voice-voxceleb_angle_proto/so-ted-portuguese-without-interpolation"
MODEL_PATH = MODEL_RUN_PATH + "320k.pth.tar"
CONFIG_PATH = MODEL_RUN_PATH + "config.json"
USE_CUDA = True
'''
DATASETS_NAME = ['brspeech', 'brspeech'] # list the datasets
DATASETS_PATH = [ '../../../../datasets/BRSpeech-3-Speakers-Paper/TTS-Portuguese_Corpus/', '../../../../datasets/BRSpeech-3-Speakers-Paper/TTS-Portuguese_Corpus/']

DATASETS_METAFILE = [ 'train_TTS-Portuguese_Corpus_metadata.csv','test_TTS-Portuguese_Corpus_metadata.csv']

#DATASETS_NAME = ['brspeech'] # list the datasets
#DATASETS_PATH = ['../../../../datasets/BRSpeech-3-Speakers-Paper/TTS-Portuguese_Corpus/']

#DATASETS_METAFILE = ['test_TTS-Portuguese_Corpus_metadata.csv']



# In[ ]:


#Preprocess dataset
meta_data = []
for i in range(len(DATASETS_NAME)):
    preprocessor = importlib.import_module('TTS.tts.datasets.preprocess')
    preprocessor = getattr(preprocessor, DATASETS_NAME[i].lower())
    meta_data += preprocessor(DATASETS_PATH[i],DATASETS_METAFILE[i])
      
meta_data= list(meta_data)'''

files =  glob('../../../../datasets/TED-100-speakers-paper/TED_Sample/**/*.wav', recursive=True)
print()
meta_ted = []
ted_speakers = set()
for i in tqdm(range(len(files))):
    if len(ted_speakers) >= 109:
        continue
    wav_file = files[i]
    speaker_id = wav_file.split('/')[-3]
    if speaker_id not in ted_speakers:
        print(speaker_id)
    meta_ted.append(['', wav_file, speaker_id])
    ted_speakers.add(speaker_id)

meta_data= meta_ted
print("TED Speakers:", len(ted_speakers))
# In[ ]:
c = load_config(CONFIG_PATH)
ap = AudioProcessor(**c['audio'])

#TTS_sample_rate = 22050
TTS_sample_rate = ap.sample_rate
SE_sample_rate = ap.sample_rate
ap.sample_rate = TTS_sample_rate
print(SE_sample_rate, TTS_sample_rate)
model = SpeakerEncoder(**c.model)
model.load_state_dict(torch.load(MODEL_PATH)['model'])
model.eval()
if USE_CUDA:
    model.cuda()

embeddings_dict = {}
len_meta_data = len(meta_data)

for i in tqdm(range(len_meta_data)):
    _, wav_file, speaker_id = meta_data[i]
    wav_file_name = os.path.basename(wav_file)
    mel_spec = ap.melspectrogram(ap.load_wav(wav_file, sr=TTS_sample_rate)).T
    mel_spec = torch.FloatTensor(mel_spec[None, :, :])
    if USE_CUDA:
        mel_spec = mel_spec.cuda()
    try:
        embedd = model.compute_embedding(mel_spec, model_sr=SE_sample_rate, spec_sr=TTS_sample_rate).cpu().detach().numpy().reshape(-1)
    except:
        print("sample ignored: ", wav_file)
        continue
    embeddings_dict[wav_file_name] = [embedd, speaker_id]


# In[ ]:


# create and export speakers.json

os.makedirs(JSON_OUTPUT,exist_ok=True)
speaker_mapping = {sample: {'name': embeddings_dict[sample][1], 'embedding':embeddings_dict[sample][0].reshape(-1).tolist()} for i, sample in enumerate(embeddings_dict.keys())}
save_speaker_mapping(JSON_OUTPUT, speaker_mapping)


# In[ ]:


#test load integrity
speaker_mapping_load = load_speaker_mapping(JSON_OUTPUT)
assert speaker_mapping == speaker_mapping_load
print("The file speakers.json has been exported to ",JSON_OUTPUT, ' with ', len(embeddings_dict.keys()), ' speakers')

