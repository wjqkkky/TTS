import argparse
import io
import json
import logging
import os
import string
import time

import numpy as np
import torch
import torchaudio

from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, _phonemes, _symbols
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from wavegrad_vocoder.inference import load_model, predict

fh = logging.FileHandler(encoding='utf-8', mode='a', filename="log/tts.log")
logging.basicConfig(level=logging.INFO, handlers=[fh], format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Synthesizer:
	def __init__(self):
		self.speaker_embedding_dict = {}
		self.speaker_names = set()
		self.speaker_embedding_dim = 0
		self.taco_model = None
		self.wg_model = None
		self.ap = None
		self.C = None

	def load(self, taco_checkpoint, wg_checkpoint, ebd_file_path, config_path, noise_schedule):
		# load the config
		self.C = load_config(config_path)
		self.C.forward_attn_mask = True

		# load the audio processor
		self.ap = AudioProcessor(**self.C.audio)

		# if the vocabulary was passed, replace the default
		if 'characters' in self.C.keys():
			symbols, phonemes = make_symbols(**self.C.characters)
		# load speakers
		logger.info(" > Start loading speaker embedding ...")
		start_time = time.time()
		speaker_mapping = json.load(open(ebd_file_path, 'r'))
		self.speaker_embedding_dim = len(speaker_mapping[list(speaker_mapping.keys())[0]]['embedding'])
		num_speakers = len(speaker_mapping)
		for key in list(speaker_mapping.keys()):
			self.speaker_names.add(speaker_mapping[key]['name'])
		for speaker_name in self.speaker_names:
			self.speaker_embedding_dict[speaker_name] = []
		for key in list(speaker_mapping.keys()):
			cur_speaker = speaker_mapping[key]['name']
			self.speaker_embedding_dict[cur_speaker].append(speaker_mapping[key]['embedding'])
		# takes the average of the embedings samples of the announcers
		for speaker in self.speaker_embedding_dict:
			self.speaker_embedding_dict[speaker] = np.mean(np.array(self.speaker_embedding_dict[speaker]),
			                                               axis=0).tolist()
		time_consuming = time.time() - start_time
		logger.info(" > Complete, time consuming {}s".format(round(time_consuming, 2)))
		logger.info(" > Start loading Taco2 ...")
		start_time = time.time()
		num_chars = len(_phonemes) if self.C.use_phonemes else len(_symbols)
		self.taco_model = setup_model(num_chars, num_speakers, self.C, self.speaker_embedding_dim)
		cp = torch.load(taco_checkpoint, map_location=torch.device('cpu'))
		self.taco_model.load_state_dict(cp['model'])
		self.taco_model.eval()
		self.taco_model.cuda()
		self.taco_model.decoder.set_r(cp['r'])
		time_consuming = time.time() - start_time
		logger.info(" > Complete, time consuming {}s".format(round(time_consuming, 2)))
		logger.info(" > Start loading wavegrad ...")
		start_time = time.time()
		params = {}
		if noise_schedule:
			params['noise_schedule'] = noise_schedule
		self.wg_model = load_model(model_dir=wg_checkpoint, params=params)
		time_consuming = time.time() - start_time
		logger.info(" > Load wavegrad model, time consuming {}s".format(round(time_consuming, 2)))

	def synthesize(self, text, speaker_name):
		t_1 = time.time()
		_, _, _, mel_postnet_spec, _, _ = synthesis(self.taco_model, text, self.C, True, self.ap, None, None,
		                                            False, self.C.enable_eos_bos_chars, False, False,
		                                            speaker_embedding=self.speaker_embedding_dict[speaker_name])
		t_2 = time.time()
		logger.info(" > Taco complete, time consuming {}s".format(round(t_2 - t_1, 2)))
		audio, sr = predict(torch.tensor(mel_postnet_spec.T), self.wg_model)
		t_3 = time.time()
		logger.info(" > Wavegrad complete, time consuming {}s".format(round(t_3 - t_2, 2)))
		audio = audio.cpu().numpy().squeeze()
		rtf = (t_3 - t_1) / (len(audio) / self.ap.sample_rate)
		logger.info(" > Real-time factor: {}".format(rtf))
		return audio
