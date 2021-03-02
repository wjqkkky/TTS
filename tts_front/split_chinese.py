# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
def split_text(text):
	# TODO 切割后的text不包含！？
	pattern_str = "，|：|。|；|？|！|——"
	match = re.search(r"\W", text)
	if not match:
		return [text]
	res = []
	texts = re.split(pattern_str, text)
	if texts[-1] == "":
		texts = texts[:-1]
	cur_text = ""
	for text in texts:
		if len(cur_text) == 0:
			cur_text += text
		else:
			cur_text += "，" + text
		if len(cur_text) > 10:
			res.append(cur_text + "。")
			cur_text = ""
	if cur_text != "":
		res.append(cur_text + "。")
	return res
