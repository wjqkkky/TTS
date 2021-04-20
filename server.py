import argparse
import datetime
import io
import logging
import logging.config
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import tornado.escape
import tornado.ioloop
import tornado.log
import tornado.options
import tornado.options
import tornado.web
import yaml
from scipy.io import wavfile
from tornado import gen
from tornado.concurrent import run_on_executor

from TTS.server.live_synthesizer import Synthesizer
from tts_front.ChineseRhythmPredictor.models.bilstm_cbow_pred_jiang_test_haitian import BiLSTM
from tts_front.synthesize_rhythm import ModelRhythm
from tts_front.tts_main import main

html_body = '''<html><title>TTS Demo</title><meta charset='utf-8'>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <textarea id="text" type="text" style="width:400px; height:200px;" placeholder="请输入要合成的文字..."></textarea>
  <script>
  document.getElementById("text").value="由汽车之家举办的第二届八幺八全球超级车展8月27日落幕。这场持续一个月的网上车展开设了近百个独立品牌展馆，推出了百城车展、全球汽车夜、汽车新消费论坛、金融节等主题活动。从公开数据看，车展累计浏览独立用户数超过1.3亿，视频播放2.7亿次。活动不仅聚焦购车，还涵盖跨界营销、汽车文化、行业交流、技术展览等多元内涵，成为汽车行业营销的新“IP”。"
  </script> 
  <button id="button" name="synthesize">合成</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
        q('#message').textContent = '合成中...'
        q('#button').disabled = true
        q('#audio').hidden = true
        synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/qicheren/synthesize?text=' + encodeURIComponent(text) + '&voice=' + getQueryVariable("voice") + '&mode=' + getQueryVariable("mode"), {cache: 'no-cache'})
        .then(function(res) {
          if (!res.ok) throw Error(res.statusText)
          return res.blob()
        }).then(function(blob) {
          q('#message').textContent = ''
          q('#button').disabled = false
          q('#audio').src = URL.createObjectURL(blob)
          q('#audio').hidden = false
        }).catch(function(err) {
          q('#message').textContent = 'ERROR! '
          q('#button').disabled = false
        })
}
function getQueryVariable(variable)
{
       var query = window.location.search.substring(1);
       var vars = query.split("&");
       for (var i=0;i<vars.length;i++) {
               var pair = vars[i].split("=");
               if(pair[0] == variable){return pair[1];}
       }
       return(1);
}
</script></body></html>
'''
## get TF logging
# logging = logging.getLogger(__name__)
# for handler in logging.handlers:
#    print(handler)
#    logging.removeHandler(handler)
# logging.setLevel(logging.DEBUG)
## create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
## create file handler which logs even debug messages
# fh = logging.FileHandler(encoding='utf-8', mode='a', filename='log/tts.log')
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logging.addHandler(fh)
speakers_dic = {1: "haitian031", 2: "m2voc_S1female", 3: "niuman"}


class MainHandler(tornado.web.RequestHandler, object):
    def get(self):
        self.set_header("Content-Type", "text/html")
        self.write(html_body)


class SynHandler(tornado.web.RequestHandler, object):
    executor = ThreadPoolExecutor(1)

    @gen.coroutine
    def get(self):
        try:
            orig_text = self.get_argument('text')
            logging.info("Receiving get request - [%s]", orig_text)
            mode = self.get_argument("mode", None, True)
            if mode:
                mode = int(mode)
                assert mode in [1, 2]
            else:
                mode = 1
            voice = self.get_argument("voice", None, True)
            if voice:
                voice = int(voice)
            else:
                voice = 1
            speaker = speakers_dic[voice]
            pcm_arr = yield self.syn(orig_text, mode, speaker)
            audio_stream = norm(pcm_arr)
            self.set_header("Content-Type", "audio/wav")
            self.write(audio_stream.getvalue())
        except Exception as e:
            logging.exception(e)

    @gen.coroutine
    def post(self):
        res = {}
        speaker = ""
        try:
            body_json = tornado.escape.json_decode(self.request.body)
            text = body_json["text"]
            logging.info("Receiving post request - [%s]", text)
            mode = self.get_argument("mode", None, True)
            if mode:
                mode = int(mode)
                assert mode in [1, 2]
            else:
                mode = 1
            voice = self.get_argument("voice", None, True)
            if voice:
                voice = int(voice)
                if voice not in speakers_dic.keys():
                    res["returnCode"] = 103
                    res["message"] = "Voice ID Error"
                    self.finish(tornado.escape.json_encode(res))
            else:
                voice = 1
            speaker = speakers_dic[voice]
        except Exception as e:
            self.set_header("Content-Type", "text/json;charset=UTF-8")
            logging.exception(e)
            res["returnCode"] = 101
            res["message"] = "Param Error"
            self.finish(tornado.escape.json_encode(res))
            return
        try:
            pcm_arr = yield self.syn(text, mode, speaker)
            audio_stream = norm(pcm_arr)
            self.set_header("Content-Type", "audio/wav")
            self.finish(audio_stream.getvalue())
        except Exception as e:
            self.set_header("Content-Type", "text/json;charset=UTF-8")
            logging.exception(e)
            res["returnCode"] = 102
            res["message"] = "Internal Server Error"
            self.finish(tornado.escape.json_encode(res))

    @run_on_executor
    def syn(self, text, mode=1, speaker="haitian031"):
        """
        inference audio
        :param text:
        :param mode: 1，正常模式，文本会过前端转成音素；2，测试模式，不过前端
        :param speaker 说话人
        :return:
        """
        pcms = np.array([])
        if mode == 1:
            start_time = datetime.datetime.now()
            ch_rhy_list, phone_list = split_text(text.strip())
            end_time = datetime.datetime.now()
            period = round((end_time - start_time).total_seconds(), 3)
            logging.info("Front-end split result: %s, %s. Time consuming: [%sms]", ch_rhy_list, phone_list,
                         period * 1000)
            sentence_num = len(ch_rhy_list)
            start_time = datetime.datetime.now()
            for i in range(sentence_num):
                # cur_sentence = ch_rhy_list[i]
                cur_phones = phone_list[i]
                res = synth.synthesize(cur_phones, speaker)
                pcms = concat_pcm_arr(pcms, res)
            end_time = datetime.datetime.now()
            period = round((end_time - start_time).total_seconds(), 3)
            logging.info("Complete, Time consuming: [%sms]", period * 1000)
        elif mode == 2:
            start_time = datetime.datetime.now()
            res = synth.synthesize(text, speaker)
            end_time = datetime.datetime.now()
            period = round((end_time - start_time).total_seconds(), 3)
            pcms = concat_pcm_arr(pcms, res)
            logging.info("Complete, Time consuming: [%sms]", period * 1000)
        else:
            raise Exception("Unknown mode : {}".format(mode))
        return pcms.copy()


def concat_pcm_arr(arr_1, arr_2):
    arr_tup = (arr_1.copy(), np.zeros(4000, dtype=np.float32), arr_2.copy())
    new_pcm_arr = np.concatenate(arr_tup)
    return new_pcm_arr


def split_text(text):
    ch_rhy_list, phone_list = main(text, rhy_model)
    return ch_rhy_list, phone_list


def norm(pcm_arr):
    audio_stream = io.BytesIO()
    wav_norm = pcm_arr * (32767 / max(0.01, np.max(np.abs(pcm_arr))))
    wavfile.write(audio_stream, 22050, wav_norm.astype(np.int16))
    return audio_stream


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--taco_model', default='../model/checkpoint_70000.pth.tar', help='Path to taco checkpoint')
    parser.add_argument('--wavegrad_model', default='../model/weights-788040.pt',
                        help='Path to wavegrad checkpoint')
    parser.add_argument('--ebd_file', default='../model/speakers_all.json',
                        help='Path to speaker embedding file')
    parser.add_argument('--config', default='config_edresson.json',
                        help='Path to speaker embedding file')
    parser.add_argument('--noise_schedule', default="../model/mult_10_788040_noise_schedule.npy",
                        help='Noise_schedule npy filepath')
    parser.add_argument('--port', default=16006, help='Port of Http service')
    parser.add_argument('--host', default="0.0.0.0", help='Host of Http service')
    parser.add_argument('--fraction', default=0.5, help='Usage rate of per GPU.')
    parser.add_argument('--frontend_mode', default=1, help='Usage rate of per GPU.')

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    taco_model_path = os.path.join(args.taco_model)
    wavegrad_model_path = os.path.join(args.wavegrad_model)
    ebd_file_path = os.path.join(args.ebd_file)
    config_path = os.path.join(args.config)
    noise_schedule = None
    if args.noise_schedule:
        noise_schedule = torch.from_numpy(np.load(args.noise_schedule))
    model = 'end'
    key_model = int(args.frontend_mode)
    if key_model == 1:
        rhy_model = 'end'
    elif key_model == 2:
        rhy_model = 'None'
    elif key_model == 3:
        rhy_model = ModelRhythm()
    else:
        raise Exception('key_model val error')

    try:
        gpu_memory_fraction = float(args.fraction)
        synth = Synthesizer()
        synth.load(taco_model_path, wavegrad_model_path, ebd_file_path, config_path, noise_schedule)
    except Exception as e:
        logging.exception(e)
    logging.config.dictConfig(yaml.load(open('logging.yaml', 'r')))
    logging.info("TTS service started...")
    # print(logging.handlers)
    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"//synthesize", SynHandler),
    ])

    application.listen(int(args.port), xheaders=True)
    tornado.ioloop.IOLoop.instance().start()
