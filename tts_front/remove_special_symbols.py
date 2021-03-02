# -*- coding: UTF-8 -*-

FH_SPACE = FHS = ((u"　", u" "),)
FH_NUM = FHN = (
	(u"０", u"0"), (u"１", u"1"), (u"２", u"2"), (u"３", u"3"), (u"４", u"4"),
	(u"５", u"5"), (u"６", u"6"), (u"７", u"7"), (u"８", u"8"), (u"９", u"9"),
)
FH_ALPHA = FHA = (
	(u"ａ", u"a"), (u"ｂ", u"b"), (u"ｃ", u"c"), (u"ｄ", u"d"), (u"ｅ", u"e"),
	(u"ｆ", u"f"), (u"ｇ", u"g"), (u"ｈ", u"h"), (u"ｉ", u"i"), (u"ｊ", u"j"),
	(u"ｋ", u"k"), (u"ｌ", u"l"), (u"ｍ", u"m"), (u"ｎ", u"n"), (u"ｏ", u"o"),
	(u"ｐ", u"p"), (u"ｑ", u"q"), (u"ｒ", u"r"), (u"ｓ", u"s"), (u"ｔ", u"t"),
	(u"ｕ", u"u"), (u"ｖ", u"v"), (u"ｗ", u"w"), (u"ｘ", u"x"), (u"ｙ", u"y"), (u"ｚ", u"z"),
	(u"Ａ", u"A"), (u"Ｂ", u"B"), (u"Ｃ", u"C"), (u"Ｄ", u"D"), (u"Ｅ", u"E"),
	(u"Ｆ", u"F"), (u"Ｇ", u"G"), (u"Ｈ", u"H"), (u"Ｉ", u"I"), (u"Ｊ", u"J"),
	(u"Ｋ", u"K"), (u"Ｌ", u"L"), (u"Ｍ", u"M"), (u"Ｎ", u"N"), (u"Ｏ", u"O"),
	(u"Ｐ", u"P"), (u"Ｑ", u"Q"), (u"Ｒ", u"R"), (u"Ｓ", u"S"), (u"Ｔ", u"T"),
	(u"Ｕ", u"U"), (u"Ｖ", u"V"), (u"Ｗ", u"W"), (u"Ｘ", u"X"), (u"Ｙ", u"Y"), (u"Ｚ", u"Z"),
)
FH_PUNCTUATION = FHP = (
	(u"．", u"."), (u"，", u","), (u"！", u"!"), (u"？", u"?"), (u"”", u'"'),
	(u"’", u"'"), (u"‘", u"`"), (u"＠", u"@"), (u"＿", u"_"), (u"：", u":"),
	(u"；", u";"), (u"＃", u"#"), (u"＄", u"$"), (u"％", u"%"), (u"＆", u"&"),
	(u"（", u"("), (u"）", u")"), (u"‐", u"-"), (u"＝", u"="), (u"＊", u"*"),
	(u"＋", u"+"), (u"－", u"-"), (u"／", u"/"), (u"＜", u"<"), (u"＞", u">"),
	(u"［", u"["), (u"￥", u"\\"), (u"］", u"]"), (u"＾", u"^"), (u"｛", u"{"),
	(u"｜", u"|"), (u"｝", u"}"), (u"～", u"~"),(u"。", u"."),
)
FH_ASCII = HAC = lambda: ((fr, to) for m in (FH_ALPHA, FH_NUM, FH_PUNCTUATION) for fr, to in m)

HF_SPACE = HFS = ((u" ", u"　"),)
HF_NUM = HFN = lambda: ((h, z) for z, h in FH_NUM)
HF_ALPHA = HFA = lambda: ((h, z) for z, h in FH_ALPHA)
HF_PUNCTUATION = HFP = lambda: ((h, z) for z, h in FH_PUNCTUATION)
HF_ASCII = ZAC = lambda: ((h, z) for z, h in FH_ASCII())


def convert(text, *maps, **ops):
	""" 全角/半角转换
	args:
		text: unicode string need to convert
		maps: conversion maps
		skip: skip out of character. In a tuple or string
		return: converted unicode string
	"""

	# if "skip" in ops:
	# 	skip = ops["skip"]
	# 	if isinstance(skip, basestring):
	# 		skip = tuple(skip)
	#
	# 	def replace(text, fr, to):
	# 		return text if fr in skip else text.replace(fr, to)
	# else:
	# 	def replace(text, fr, to):
	# 		return text.replace(fr, to)

	for m in maps:
		if callable(m):
			m = m()
		elif isinstance(m, dict):
			m = m.items()
		for fr, to in m:
			text = replace(text, fr, to)
	return text

"""判断一个unicode是否是汉字"""
from idna import unichr, unicode

def is_chinese(uchar):
	if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
		return True
	else:
		return False


"""判断一个unicode是否是数字"""


def is_number(uchar):
	if uchar >= u'\u0030' and uchar <= u'\u0039':
		return True
	else:
		return False


"""判断一个unicode是否是英文字母"""


def is_alphabet(uchar):
	if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
		return True
	else:
		return False


"""判断是否是（汉字，数字和英文字符之外的）其他字符"""


def is_other(uchar):
	if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
		return True
	else:
		return False


"""半角转全角"""


def B2Q(uchar):
	rstring = ""
	inside_code = ord(uchar)
	if inside_code == 12288:  # 全角空格直接转换
		inside_code = 32
	elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
		inside_code -= 65248
	rstring += chr(inside_code)
	return rstring


"""全角转半角"""


def Q2B(uchar):
	'''
	全角转半角，可转换的字符包括；（，？！；：）
	:param uchar: 单个字符，
	:return: 转化后的字符
	'''
	inside_code = ord(uchar)
	if inside_code == 0x3000:
		inside_code = 0x0020
	else:
		inside_code -= 0xfee0
	if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
		return uchar
	return unichr(inside_code)

def strQ2B(ustring):
	"""全角转半角"""
	rstring = ""
	for uchar in ustring:
		inside_code=ord(uchar)
		if inside_code == 12288:                              #全角空格直接转换
			inside_code = 32
		elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
			inside_code -= 65248

		rstring += unichr(inside_code)
	return rstring
"""把字符串全角转半角"""

def is_reserved_symbols(uchar):
	inside_code = ord(uchar)
	list_symbols =[12289,12290,44,46,59,58,33,63,42]
	if inside_code in list_symbols:
		return unichr(inside_code)
	else:
		return 0

def stringQ2B(ustring):
	return "".join([Q2B(uchar) for uchar in ustring])


"""将UTF-8编码转换为Unicode编码"""


def convert_toUnicode(string):
	ustring = string
	if not isinstance(string, unicode):
		ustring = string.decode('UTF-8')
	return ustring

def remove_symbols(mandarin):
	'''
	去除特殊符号，仅仅保留汉字、数字、字母、特殊符号[（。,?!:;）]
	:param mandarin: 需要去除特殊符号的字符串
	:return: 去除特殊符号后的字符串
	'''
	# mandarin = "****（）——！23@#￥%……%GTHMa啊啊。/；‘；’21：34"
	new_mandarin = ''
	for char in mandarin.strip():
		char_uni = convert_toUnicode(char)
		if is_chinese(char_uni) or is_number(char_uni) or is_alphabet(char_uni):
			new_mandarin = new_mandarin + char_uni
		else:
			char_Q2B = Q2B(char_uni)
			if is_reserved_symbols(char_Q2B):
				new_mandarin = new_mandarin + is_reserved_symbols(char_Q2B)
	return new_mandarin

# if __name__ == "__main__":
#
# 	ustring1 = u'收割季节 麦浪和月光 洗着快镰刀'
# 	string1 = 'Sky0天地Earth1*，que。'
#
# 	ustring1 = convert_toUnicode(ustring1)
# 	string1 = convert_toUnicode(string1)
#
# 	for item in string1:
# 		# print is_chinese(item)
# 		# print is_number(item)
# 		# print is_alphabet(item)
# 		if is_other(item):
# 			print(item)

if __name__ == '__main__':
	# mandarin = "****（）——！、23@#￥%……%GTHMa啊啊。/；‘；’21：34"
	# new_mandarin = ''
	# for char in mandarin.strip():
	# 	char_uni = convert_toUnicode(char)
	# 	if is_chinese(char_uni) or is_number(char_uni) or is_alphabet(char_uni):
	# 		new_mandarin = new_mandarin+char_uni
	# 	else:
	# 		char_Q2B = Q2B(char_uni)
	# 		if is_reserved_symbols(char_Q2B):
	# 			new_mandarin = new_mandarin+char_uni
	# print(new_mandarin)
	print(ord("*"))
