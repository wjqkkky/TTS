# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
from tts_front.chinese_to_tn import *
import tts_front.remove_special_symbols as rm
# from tts_front.en2phoneme import word2phone
import re
import tts_front.synthesize_rhythm as synth_rhy
from tts_front.split_chinese import split_text
from tts_front.grapheme_to_phoneme import Grapheme2phoneme

# from memory_profiler import profile  # 检查内存占用情况


G2p = Grapheme2phoneme()


def regula_specail(chinese: str):
    """
    匹配特殊符号
    :param chinese:汉字字符串
    :return: 正则化后的字符串
    """
    # ##正则化匹配网址
    # regular = re.compile(r'[a-zA-z]+://[^\s]*')
    regular = re.compile(
        r'([hH][tT]{2}[pP]://|[hH][tT]{2}[pP][sS]://|[wW]{3}.|[wW][aA][pP].|[fF][tT][pP].|[fF][iI][lL][eE].)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')
    chinese = re.sub(pattern=regular, repl="", string=chinese)
    # #约等号规则：约等号前后为数字，
    # pattern = re.compile(r'(\d+≈\d)')#(r'(\d+( )+?≈( )+?\d)')
    pattern = re.compile(r'(\d+( )?≈( )?\d+)')
    matchers = pattern.findall(chinese)
    for matcher in matchers:
        chinese = chinese.replace(matcher[0], matcher[0].replace('≈', '约等于'))
    # #波浪号规则：波浪号前后为数字，
    pattern = re.compile(r'(\d+( )?~( )?\d)')
    matchers = pattern.findall(chinese)
    for matcher in matchers:
        chinese = chinese.replace(matcher[0], matcher[0].replace('~', '至'))
    # ###将'] '、’) ‘,’） ‘，反括号空格替换为逗号，反括号加空格的格式
    pattern = '\) |\] |\} |\） '
    # chinese = re.sub(pattern=pattern, repl=",", string=chinese)
    chinese = re.sub(pattern=pattern, repl="、", string=chinese)
    # ##正则化匹配(数字)、(1).规则：将反括号前为数字的情况，全部替换为数字加顿号。例如：1）换为1,，（十一）换成十一、
    # 注意：括号匹配前必须将中文括号转化为英文括号
    # regular = re.compile(r'[\( *?\)]')
    # regular = r'[(][(.*?)+[)]'#匹配括号加括号内的
    regular = r'(.*?)[\)]'  # 反括号内的内容
    # regular_in =r'[(](.*?)[)]'#匹配括号内的内容
    parentheses_list = re.findall(regular, chinese)
    number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                   '零']
    for parenthese in parentheses_list:
        if parenthese[-1] in number_list:
            # chinese = chinese.replace(parenthese[-1] + ')', parenthese[-1] + ',').replace('(', '')
            chinese = chinese.replace(parenthese[-1] + ')', parenthese[-1] + '、').replace('(', '')
    # chinese = chinese.replace('(', '“').replace(')', '“')#将（）左右括号转化为#1
    chinese = chinese.replace('(', '“').replace(')', '')#将“（”转化为#1，“）”省略
    # chinese = chinese.replace('(', ',').replace(')', ',')
    # ###正则化匹配“第*条|章 ”规则，匹配规则：只要为"第***+空格"替换为"第***,"***字符串的个数为3
    # regular = r'[第]+(.*?)+[章 ]'
    # regular = r'([第]+(.*?)+[ ])'
    regular = r'([第](.*?)[ ]+\b)'
    parentheses_list = re.findall(regular, chinese)

    for parenthese in parentheses_list:
        if parenthese[0] != '' and len(parenthese[0]) < 6:
            # chinese = chinese.replace(parenthese[0], parenthese[0][:-1] + ',')
            chinese = chinese.replace(parenthese[0], parenthese[0][:-1] + '、')
    return chinese


def is_continuous_sign(sign):
    """
    检查整个字符串是否包含中文,字母，数字。若含有中文、字母、数字则返回True；若不含则返回false
    :param sign: 需要检查的字符串
    :return: bool
    """
    sign = sign.encode('utf-8').decode('utf-8')
    for ch in sign:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
        if (u'\u0041' <= ch <= u'\u005a') or (u'\u0061' <= ch <= u'\u007a'):
            return True
        if ch.isdigit():
            return True
    return False


def rm_continuous_sign(chinese: str):
    """
    去除连续字符,例如连续的逗号和连续的句号，连续的标点符号
    :return:
    :param chinese:去除前的 字符串
    :return: 去除后的字符串
    """
    pattern = ',|。|，'
    chinese_list = re.split(pattern=pattern, string=chinese)
    chinese = ''
    split_chinese_list = []  # 返回列表
    flag_begin = 1
    for chinese_ in chinese_list:
        if chinese_ == '':
            continue
        if not is_continuous_sign(chinese_):  # 若chinese_没有单词字母和数字则忽略
            continue
        if flag_begin:
            chinese = chinese + chinese_
            flag_begin = 0
        else:
            chinese = chinese + ',' + chinese_
        split_chinese_list.append(chinese_ + ',')
    if len(split_chinese_list) > 1:
        split_chinese_list[-1] = split_chinese_list[-1][:-1] + '。'
    return chinese+'。', split_chinese_list


# @profile
def main(chinese: str, model: object):
    chinese = chinese.replace('（', '(').replace('）', ')').replace('\n', '。').replace('\r', '。').replace('\t', '。')
    # ##匹配url，第*章 ，第* ，（反括号前为数字），反括号前为数字）
    chinese = regula_specail(chinese)
    # ###多个连续空格保留一个，若最后一个为空格则去除
    chinese = ' '.join(chinese.split())
    if len(chinese) == 0:  # 若输入的有效字符为空，则输出为空
        return [''], ['']
    # if chinese[-1] == ' ':
    #     chinese = chinese[:-1]
    # ###将空格替换为* 必须替换，否则会模型预测时，会将空格给替换为空
    chinese = chinese.replace(' ', '*')
    chinese_list = split_text(chinese)
    ch_rhy_list = []
    phone_list = []
    for ch in chinese_list:
        chinese_nor = NSWNormalizer(ch).normalize()
        chinese_nor = chinese_nor.replace(":", ',').replace('.', '。').replace('-', ',').replace('、', ',')
        chinese_rm_symbols = rm.remove_symbols(chinese_nor)
        # chinese_rm_symbols为str，chinese_rm_symbols_list为列表
        chinese_rm_symbols, chinese_rm_symbols_list = rm_continuous_sign(chinese_rm_symbols)
        if model == 'end':
            chinese_rhy = synth_rhy.ending_add_rhy(chinese_rm_symbols)
        elif model == 'None':
            chinese_rhy = chinese_rm_symbols
        else:
            chinese_rhy = model.model_rhy(chinese_rm_symbols_list, chinese_rm_symbols)
        if "“" in chinese_rhy and model != 'None':
            chinese_rhy = rm.replace_first_rhy(chinese_rhy, sig='“')
        # chinese2phone = word2phone(chinese_rhy, chinese_split=True, chinese_u2v=True)
        chinese2phone = G2p.trans2phone(chinese_rhy)
        chinese2phone = chinese2phone.replace('*', ' ')
        # ###去除空格，多个空格保留一个
        chinese2phone = ' '.join(chinese2phone.split())
        if len(chinese2phone) == 0:
            continue
        if chinese2phone[-1].isdigit():
            chinese2phone = chinese2phone + ' .'
        if model != 'None':
            chinese_rhy = chinese_rhy.strip()
            chinese_rhy = chinese_rhy[:-2] + '3' + chinese_rhy[-1]
            chinese2phone = chinese2phone[:-3] + '3 .'
            chinese_rhy = chinese_rhy.replace(',', '').replace('，', '').replace('.', '').replace('。', '')
            chinese2phone = chinese2phone.replace(' ,', '').replace(' .', '')
        ch_rhy_list.append(chinese_rhy)
        phone_list.append(chinese2phone)
    return ch_rhy_list, phone_list
