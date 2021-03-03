import argparse
import base64
import datetime
import io
import json
import logging
import os
import re
import traceback
import uuid
import numpy as np
import tornado.web
import tornado.ioloop
import tornado.escape
import tornado.options
import tornado.log
from scipy.io import wavfile
from tornado import gen
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

from TTS.server.live_synthesizer import Synthesizer
from tts_front.ChineseRhythmPredictor.models.bilstm_cbow_pred_jiang_test_haitian import BiLSTM
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
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
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
</script></body></html>
'''

use_options = tornado.options.options
use_options.log_to_stderr = True
# use_options.log_rotate_mode = str('time')
# use_options.log_file_prefix = str('./log/tts_server.log')
# use_options.log_rotate_when = str('W0')
# use_options.log_rotate_interval = 2
fh = logging.FileHandler(encoding='utf-8', mode='a', filename="log/tts.log")
logging.basicConfig(level=logging.INFO, handlers=[fh], format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
tornado.log.enable_pretty_logging(use_options)
speakers_dic = {1: "haitian031", 2: "niuman", 3: "m2voc_S1female"}


class MainHandler(tornado.web.RequestHandler, object):
	def get(self):
		self.set_header("Content-Type", "text/html")
		self.write(html_body)


class SynHandler(tornado.web.RequestHandler, object):
	executor = ThreadPoolExecutor(15)

	@gen.coroutine
	def get(self):
		try:
			orig_text = self.get_argument('text')
			logger.info("Receiving get request - [%s]", orig_text)
			mode = self.get_argument("mode", None, True)
			if mode:
				mode = int(mode)
				assert mode in [0, 1]
			else:
				mode = 0
			voice = self.get_argument("voice", None, True)
			if voice:
				voice = int(voice)
			else:
				voice = 0
			speaker = speakers_dic[voice]
			pcms = yield self.syn(orig_text, mode, speaker)
			wav = io.BytesIO()
			wavfile.write(wav, 22050, pcms.astype(np.int16))
			self.set_header("Content-Type", "audio/wav")
			self.write(wav.getvalue())
		except Exception as e:
			logger.exception(e)

	@gen.coroutine
	def post(self):
		res = {}
		speaker = ""
		try:
			body_json = tornado.escape.json_decode(self.request.body)
			text = body_json["text"]
			mode = self.get_argument("mode", None, True)
			if mode:
				mode = int(mode)
				assert mode in [0, 1]
			else:
				mode = 0
			voice = self.get_argument("voice", None, True)
			if voice:
				voice = int(voice)
				if voice not in speakers_dic.keys():
					res["returnCode"] = 103
					res["message"] = "Voice ID Error"
					self.finish(tornado.escape.json_encode(res))
			else:
				voice = 0
			speaker = speakers_dic[voice]
		except Exception as e:
			self.set_header("Content-Type", "text/json;charset=UTF-8")
			logger.exception(e)
			res["returnCode"] = 101
			res["message"] = "Param Error"
			self.finish(tornado.escape.json_encode(res))
			return
		try:
			pcms = yield self.syn(text, mode, speaker)
			logger.info("Receiving post request - [%s]", text)
			wav = io.BytesIO()
			wavfile.write(wav, 22050, pcms.astype(np.int16))
			self.set_header("Content-Type", "audio/wav")
			self.finish(wav.getvalue())
		except Exception as e:
			self.set_header("Content-Type", "text/json;charset=UTF-8")
			logger.exception(e)
			res["returnCode"] = 102
			res["message"] = "Internal Server Error"
			self.finish(tornado.escape.json_encode(res))

	@run_on_executor
	def syn(self, text, mode=0, speaker="haitian031"):
		"""
		inference audio
		:param text:
		:param mode: 0，正常模式，文本会过前端转成音素；1，测试模式，不过前端
		:param speaker 说话人
		:return:
		"""
		pcms = np.array([])
		if mode == 0:
			start_time = datetime.datetime.now()
			ch_rhy_list, phone_list = split_text(text.strip())
			end_time = datetime.datetime.now()
			period = round((end_time - start_time).total_seconds(), 3)
			logger.info("Front-end split result: %s, %s. Time consuming: [%sms]", ch_rhy_list, phone_list,
			            period * 1000)
			sentence_num = len(ch_rhy_list)
			for i in range(sentence_num):
				cur_sentence = ch_rhy_list[i]
				cur_phones = phone_list[i]
				logger.info("Inference sentence: [%s], phones: [%s]", cur_sentence, cur_phones)
				start_time = datetime.datetime.now()
				res = synth.synthesize(cur_phones, speaker)
				end_time = datetime.datetime.now()
				period = round((end_time - start_time).total_seconds(), 3)
				logger.info("Sentence total time consuming - [%sms]", period * 1000)
				pcm_arr = np.frombuffer(res, dtype=np.int16)[5000:-4000]
				pcms = np.append(pcms, pcm_arr)
		elif mode == 1:
			name = str(uuid.uuid4())
			start_time = datetime.datetime.now()
			res = synth.synthesize(text, speaker)
			end_time = datetime.datetime.now()
			period = round((end_time - start_time).total_seconds(), 3)
			logger.info("%s - sentence total time consuming - [%sms]", name, period * 1000)
			pcm_arr = np.frombuffer(res, dtype=np.int16)[5000:-4000]
			pcms = np.append(pcms, pcm_arr)
		else:
			raise Exception("Unknown mode : {}".format(mode))
		return pcms


def split_text(text):
	ch_rhy_list, phone_list = main(text, "end")
	return ch_rhy_list, phone_list


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--taco_model', default='model/tacotron_model.ckpt-175000', help='Path to taco checkpoint')
	parser.add_argument('--wavegrad_model', default='model/tacotron_model.ckpt-175000',
	                    help='Path to wavegrad checkpoint')
	parser.add_argument('--ebd_file', default='model/tacotron_model.ckpt-175000',
	                    help='Path to speaker embedding file')
	parser.add_argument('--config', default='config_edresson.json',
	                    help='Path to speaker embedding file')
	parser.add_argument('--port', default=16006, help='Port of Http service')
	parser.add_argument('--host', default="0.0.0.0", help='Host of Http service')
	parser.add_argument('--fraction', default=0.3, help='Usage rate of per GPU.')
	parser.add_argument('--frontend_mode', default=3, help='Usage rate of per GPU.')
	args = parser.parse_args()
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	taco_model_path = os.path.join(args.taco_model)
	wavegrad_model_path = os.path.join(args.wavegrad_model)
	ebd_file_path = os.path.join(args.ebd_file)
	config_path = os.path.join(args.config)
	key_model = int(args.frontend_mode)
	if key_model == 1:
		model = 'end'
	elif key_model == 2:
		model = 'None'
	else:
		model = BiLSTM()
		model.load_model()

	try:
		gpu_memory_fraction = float(args.fraction)
		synth = Synthesizer()
		synth.load(taco_model_path, wavegrad_model_path, ebd_file_path, config_path)
	except:
		traceback.print_exc()
	logger.info("TTS service started...")
	application = tornado.web.Application([
		(r"/", MainHandler),
		(r"/synthesize", SynHandler),
	])
	application.listen(int(args.port), xheaders=True)
	tornado.ioloop.IOLoop.instance().start()
