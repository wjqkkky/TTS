# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
from tts_front.change_tone import chinese_bian_diao_pinyin
from tts_front import split_phoneme
from tts_front.remove_special_symbols import *
import pronouncing
import enchant

EN_DICT = enchant.Dict("en_US")

with open('./tts_front/lexicon26.txt', 'r', encoding='utf-8') as fo:
    global GLOBAL_ENGLISH_DICT_LEXICON
    GLOBAL_ENGLISH_DICT_LEXICON = dict()
    lines = fo.readlines()
    for line in lines:
        character, phone, character_pinyin = line.split('\t')
        GLOBAL_ENGLISH_DICT_LEXICON[character] = phone


class Grapheme2phoneme:

    def __init__(self, chinese_split: object = True, chinese_u2v: object = True, read_as_alphabet: object = False) -> object:
        self.chinese_split = chinese_split
        self.chinese_u2v = chinese_u2v
        self.read_as_alphabet = read_as_alphabet
        self.zimu_read = ['a', 'it', 'led', 'mm', 'id']

    def chinese2phoneme(self, word: str):
        """
        将英文转化为音素
        :param word: 中文字符，仅仅包含中文字符，不包含标点数字以及其他的符号
        :return: 音素字符串
        """
        string_phone = chinese_bian_diao_pinyin(word)
        if self.chinese_u2v:
            string_phone = split_phoneme.u_to_v(string_phone)
        if self.chinese_split:
            string_phone = split_phoneme.split_sheng(string_phone)
        return string_phone

    def english2phoneme(self, word: str, read_as_alphabet=False):
        """
        一个单词一个单词的处理，不能处理一段英文，需要将英文以词切分开，单独输入该函数
        :param read_as_alphabet: 是否按照字符读取，
        :param word: 单词字符串
        :return: 音素字符串
        """
        flag_check = EN_DICT.check(word) or EN_DICT.check(word.title()) or EN_DICT.check(
            word.lower())  # ##True为单词，False为字母
        en_phone = ''
        if self.read_as_alphabet:
            flag_check = False
        if read_as_alphabet:
            flag_check = False
        if flag_check:
            en_phone = pronouncing.phones_for_word(word)
        if not flag_check or (0 == len(en_phone)) or (not en_phone[0].isupper()):
            word = word.upper()
            en_phone = ' / '.join([GLOBAL_ENGLISH_DICT_LEXICON[w] for w in word])
        else:
            en_phone = en_phone[0]
        return en_phone

    def trans2phone(self, grapheme:str):
        """
        将中英文切分开，单独的将英文和中文转化为音素
        :param grapheme: 原始的句子，中文
        :return:中英文分割后的列表转化为音素
        """
        list_final = split_grapheme(grapheme)
        len_list = len(list_final)
        if len_list == 0:
            return ''
        phone_final = []
        # ##将字符转化为拼音
        for i, word in enumerate(list_final):
            if word.lower() in self.zimu_read:
                if len_list < 2:
                    return [self.english2phoneme(word, read_as_alphabet=True)]
                if i == len_list - 1 and EN_DICT.check(list_final[i - 1]):
                    phone_final.append(self.english2phoneme(word))
                    continue
                if EN_DICT.check(list_final[i + 1]):
                    phone_final.append(self.english2phoneme(word))
                    continue
                phone_final.append(self.english2phoneme(word, read_as_alphabet=True))
            elif is_chinese(word[0]):  # 中文转音素
                phone_final.append(self.chinese2phoneme(word))
            elif is_alphabet(word[0]):  # 英文转音素
                phone_final.append(self.english2phoneme(word))
            else:  # 其他符号转音素
                word = word.replace(",", " ,").replace("，", " ,").replace("。", " .").replace(".", " .").replace(
                    "?", " ?").replace("？", " ?").replace("#", " #")
                phone_final.append(word)
        phoneme = en_ch_connect(phone_final)  # 将音素连接起来
        # phoneme = ' '.join(phoneme.split())
        return phoneme


def en_connect(connect_before: list):
    """
    英文连接
    :param connect_before: 连接前英文的音标
    :return:连接后字符串
    """
    count_en = 0
    for cb in connect_before:
        if cb.isupper():
            count_en = count_en + 1

    connect_after = ''
    for i in range(0, len(connect_before)):
        if count_en > 1:
            if connect_before[i].isupper():
                connect_after = connect_after + connect_before[i] + ' / '
                count_en = count_en - 1
            else:
                connect_after = connect_after + connect_before[i] + ' '
        else:
            if i < len(connect_before) - 1:
                connect_after = connect_after + connect_before[i] + ' '
            else:
                connect_after = connect_after + connect_before[i]
    return connect_after


def en_ch_connect(phone_final: list):
    """
    分别连接英文和非英文,
    :param phone_final: 中英文的音素
    :return:
    """
    flag_before, flag_now = 0, 0
    list_temp = []
    phone_each_connet = []
    others = []
    for pf in phone_final:
        if pf.isupper():
            flag_now = 1
        else:
            flag_now = 2 if pf.islower() else 0  # 英文为1，中文为2，其他为0
        if not flag_before:
            if flag_now:
                flag_before = flag_now
                list_temp.append(pf)
            continue
        if flag_now == 0:
            others.append(pf)
            continue
        if flag_now == flag_before:
            list_temp.extend(others)
            others = []  # 其他字符添加
            list_temp.append(pf)
        else:
            if flag_before == 1:  # 将英文变为字符串并添加英文
                temp_str = en_connect(list_temp)
                list_temp = others
            else:  # 添加中文，将中文变为字符串
                list_temp.extend(others)
                temp_str = ' '.join(list_temp)
                list_temp = []
            phone_each_connet.append(temp_str)
            flag_before = flag_now
            others = []
            list_temp.append(pf)
    list_temp.extend(others)
    if flag_before == 1:
        temp_str = en_connect(list_temp)
    else:
        temp_str = ' '.join(list_temp)
    phone_each_connet.append(temp_str)
    phoneme = ' / '.join(phone_each_connet)
    return phoneme


def split_grapheme(string: str):
    """
    按照英文非英文，中文非中文分开，['Xbox', ' ', 'Bar', '#1', ' ', '小 软件虽然功能不多', '#2,']
    :type string: str
    :param string:
    :return:
    """
    if string == '':
        return ''
    list_final = []
    str_word = ''
    flag, flag_begin = -1, 0
    # ##中英文分开
    for ph in string:
        flag_begin = flag_begin + 1
        flag_now = mask_word(ph)  # 1为字母，2为汉字，3为空格，4为其他
        if flag_now == flag:
            str_word = str_word + ph
        else:
            if flag_now == 3:
                if flag == 1:  # 前一个为非英文加空格，空格可以直接省略
                    flag = flag_now
                continue
            if flag_begin == 1:
                str_word = str_word + ph
            else:
                if len(str_word) > 0:
                    list_final.append(str_word)
                str_word = ph
            flag = flag_now
    list_final.append(str_word)
    return list_final


def mask_word(w: str):
    """
    将字符串进行划分，1为字母，2为汉字，3为空格，4为其他
    :param w:
    :return:
    """
    if is_alphabet(w):
        return 1
    elif is_chinese(w):
        return 2
    elif w == ' ':
        return 3
    else:
        return 4


if __name__ == '__main__':
    ch = '，，， ，，，Xbox #1 Bar #2， 小 软件虽 #1 然功能不多# 2,hello A hello#1 #1'
    G2p = Grapheme2phoneme(read_as_alphabet=True)
    # split_chinese_eng_list = split_chinese_eng(ch)
    # print(G2p.chinese2phoneme('一会儿就来'))
    # print(G2p.english2phoneme('hello'))
    # G2p.trans2phone(split_chinese_eng_list)
    # split_grapheme_list = split_grapheme(ch)
    print(ch)
    # print(split_grapheme_list)
    before_connet = G2p.trans2phone(ch)
    print(before_connet)
    # print(en_ch_connect(before_connet))
