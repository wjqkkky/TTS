#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
# pylint: disable=redefined-outer-name, unused-argument
import os
import string
import time

import numpy as np
import torch
import torchaudio

from TTS.tts.utils.synthesis import synthesis
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.utils.generic_utils import setup_generator


def tts(model, vocoder_model, text, CONFIG, use_cuda, ap, use_gl, speaker_fileid, speaker_embedding=None,
        gst_style=None):
	t_1 = time.time()
	waveform, _, _, mel_postnet_spec, _, _ = synthesis(model, text, CONFIG, use_cuda, ap, speaker_fileid, gst_style,
	                                                   False, CONFIG.enable_eos_bos_chars, use_gl,
	                                                   speaker_embedding=speaker_embedding)

	# grab spectrogram (thx to the nice guys at mozilla discourse for codesnipplet)
	if args.save_spectogram:
		spec_file_name = args.text.replace(" ", "_")[0:10]
		spec_file_name = spec_file_name.translate(
			str.maketrans('', '', string.punctuation.replace('_', ''))) + '.npy'
		spec_file_name = os.path.join(args.out_path, spec_file_name)
		spectrogram = torch.FloatTensor(mel_postnet_spec.T)
		spectrogram = spectrogram.unsqueeze(0)
		np.save(spec_file_name, spectrogram)
		print(" > Saving raw spectogram to " + spec_file_name)

	if CONFIG.model == "Tacotron" and not use_gl:
		mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T).T
	if not use_gl:
		# Use if not computed noise schedule with tune_wavegrad
		beta = np.linspace(1e-6, 0.01, 50)
		vocoder_model.compute_noise_level(beta)

		# Use alternative when using output npy file from tune_wavegrad
		# beta = np.load("output-tune-wavegrad.npy", allow_pickle=True).item()
		# vocoder_model.compute_noise_level(beta['beta'])

		device_type = "cuda" if use_cuda else "cpu"
		waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).to(device_type).unsqueeze(0))
	if use_cuda and not use_gl:
		waveform = waveform.cpu()
	if not use_gl:
		waveform = waveform.numpy()
	waveform = waveform.squeeze()
	rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
	tps = (time.time() - t_1) / len(waveform)
	print(" > Run-time: {}".format(time.time() - t_1))
	print(" > Real-time factor: {}".format(rtf))
	print(" > Time per step: {}".format(tps))
	return waveform


def synthesis(mel):
	t_1 = time.time()
	beta = np.linspace(1e-6, 0.01, 50)
	vocoder_model.compute_noise_level(beta)
	waveform = vocoder_model.inference(torch.FloatTensor(mel).to("cuda").unsqueeze(0))
	waveform = waveform.cpu().numpy()
	waveform = waveform.squeeze()
	rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
	tps = (time.time() - t_1) / len(waveform)
	print(" > Run-time: {}".format(time.time() - t_1))
	print(" > Real-time factor: {}".format(rtf))
	print(" > Time per step: {}".format(tps))
	return waveform


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'output_path',
		type=str,
		help='Path to save final wav file. Wav file will be names as the text given.',
	)
	parser.add_argument('config_path',
	                    type=str,
	                    help='Path to model config file.')
	parser.add_argument('--use_cuda',
	                    type=bool,
	                    help='Run model on CUDA.',
	                    default=False)
	parser.add_argument(
		'--vocoder_path',
		type=str,
		help=
		'Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).',
		default="",
	)
	parser.add_argument('--vocoder_config_path',
	                    type=str,
	                    help='Path to vocoder model config file.',
	                    default="")
	parser.add_argument(
		'--batched_vocoder',
		type=bool,
		help="If True, vocoder model uses faster batch processing.",
		default=True)
	parser.add_argument(
		'--spectrogram_path',
		type=str,
		help="",
		default=True)

	args = parser.parse_args()
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)

	C = load_config(args.config_path)
	C.forward_attn_mask = True

	# load the audio processor
	ap = AudioProcessor(**C.audio)

	start_time = time.time()
	# load vocoder model
	if args.vocoder_path != "":
		VC = load_config(args.vocoder_config_path)
		vocoder_model = setup_generator(VC)
		vocoder_model.load_state_dict(torch.load(args.vocoder_path, map_location="cpu")["model"])
		vocoder_model.remove_weight_norm()
		if args.use_cuda:
			vocoder_model.cuda()
		vocoder_model.eval()
	else:
		vocoder_model = None
		VC = None
	time_consuming = time.time() - start_time
	print(" > Load wavegrad model, time consuming {}s".format(round(time_consuming, 2)))
	mels = os.listdir(args.spectrogram_path)
	for mel_file in mels:
		if not os.path.isdir(mel_file) and mel_file[-2:] == "pt":
			start_time = time.time()
			print(" > Start inferencing sentence {} . ".format(mel_file))
			spectrogram = torch.tensor(torch.load(os.path.join(args.spectrogram_path, mel_file)))
			audio= synthesis(spectrogram)
			out_wav_name = os.path.join(args.output_path, mel_file[:-7] + "_wg.wav")
			ap.save_wav(audio, out_wav_name)
			time_consuming = time.time() - start_time
			print(" > Complete, time consuming {}s".format(round(time_consuming, 2)))
