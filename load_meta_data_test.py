from TTS.tts.datasets.preprocess import load_meta_data

if __name__ == '__main__':
	dataset_config = [
		{
			"name": "libri_tts",
			"path": "/data1/wangjiaqi/datasets/LibriTTS/LibriTTS/dev-clean",
			"meta_file_train": None,
			"meta_file_val": None
		},
	]

	wav_files, _ = load_meta_data(dataset_config, eval_split=False)
