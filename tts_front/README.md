# TTS-front-end-processing
需要的环境：

numpy==1.14
jieba==0.39
pandas
pypinyin==0.36.0
scikit_learn==0.21.3
torch==1.0.1.post2
tqdm==4.36.1
pronouncing
pyenchant



1.下载git clone https://github.com/ysujiang/en-tts.git

2.安装环境：
pip install -r  requirements.txt
apt-get install enchant

3.应用模型

说明：本模型训练的为二级韵律+TN处理+ChineseTone拼音转文本处理

运行：python pypi_tts_main.py

备注：在pypi_tts_main.py中，key_model=3为先LSTM加韵律，若LSTM添加失败则在结尾处添加韵律。
    key_model=1 为只在结尾处添加韵律
    key_model=2 为不添加韵律

    在tts_main.py中，chinese_split=True为声韵母切分汉语拼音，False为不切分汉语拼音
    chinese_u2v=True为将jqxy后读鱼的进行转化，false为不转换


