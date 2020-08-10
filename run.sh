python -m pip install -r requirements.txt
python setup.py develop
python TTS/bin/train_tts.py --config_path ../Jia-et-al-2018/config.json --restore_path ../Jia-et-al-2018/converted/best_model.pth.tar
