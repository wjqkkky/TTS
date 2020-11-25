from TTS.tts.datasets.preprocess import load_meta_data

if __name__ == '__main__':
	# dataset_config = [
	# 	{
	# 		"name": "libri_tts",
	# 		"path": "/data1/wangjiaqi/datasets/LibriTTS/LibriTTS/dev-clean",
	# 		"meta_file_train": None,
	# 		"meta_file_val": None
	# 	},
	# ]
	#
	# wav_files, _ = load_meta_data(dataset_config, eval_split=False)
	# print(wav_files)
	# print(_)
	dataset_config = [
		{
			"name": "aishell_3",
			"path": r"D:\data\AISHELL-3\train",
			"meta_file_train": "label.txt",
			"meta_file_val": None
		},
	]
	wav_files, _ = load_meta_data(dataset_config, eval_split=False)
	output_files = [
		wav_file[1].replace(r"D:\data\AISHELL-3\train", r"/data1/wangjiaqi/voice_clone/mozilla/output").replace(
			'.wav', '.npy') for wav_file in wav_files]

# ['Excellent salads may be made of hard eggs, or the remains of salt fish flaked nicely from the bone, by pouring over a little of the above mixture when hot, and allowing it to cool.', '/data1/wangjiaqi/datasets/LibriTTS/LibriTTS/dev-clean/1919/142785/1919_142785_000033_000001.wav', 'LTTS_1919']
	print(1)