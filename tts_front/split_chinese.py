# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
import re
def split_text_sub(text, pattern_str=','):
    # TODO 切割后的text不包含！？
    '''
    分割符号包括，[,。；\. ? ! -- \n - --]
    :param text:
    :return:
    '''
    # ，|：|。|；|？|！|——
    # pattern_str = "，|。|；|,|？|！|——|\n|-"
    # pattern_str = ",|。|;|\?|!|——|\n"
    res = []
    texts = re.split(pattern_str, text)
    if texts[-1] == "":
        texts = texts[:-1]
    cur_text = ""
    for text in texts:
        if len(cur_text) == 0:
            cur_text += text
        else:
            if text != '':
                cur_text += "，" + text
        if len(cur_text) > 10:
            cur_text = cur_text + "。"
            # cur_text = cur_text.replace('.。','。')
            res.append(cur_text)
            cur_text = ""
    if cur_text != "":
        cur_text = cur_text + "。"
        # cur_text = cur_text.replace('.。', '。')
        res.append(cur_text)
    return res


def split_text(text):
    '''
    句子第一分割符号，必须要分割的符号
    :param text:
    :return:
    '''
    text = text.replace('，', ',').replace('？', '?').replace('！', '!').replace('；', ';')
    pattern_1 = "。|\?|!|\n"
    pattern_2 = ',|;|——|……'
    match = re.search(r"\W", text)
    if not match:
        return [text]
    texts = re.split(pattern_1, text)
    res = []
    for text in texts:
        if len(text) == 0:
            continue
        if (',' in text) or (';' in text) or ('——' in text):
            sub_split = split_text_sub(text, pattern_str=pattern_2)
            res.extend(sub_split)
        else:
            res.append(text)
    return res
