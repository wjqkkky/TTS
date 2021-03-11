# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
# from nltk.corpus import words,cmudict
# from g2p_en import G2p
import pronouncing
from tts_front.change_tone import chinese_bian_diao_pinyin
from tts_front import split_phoneme
# from tts_front.tts_front_end_main import ch2py
# from tts_front.tts_front_end_main import GLOBAL_ENGLISH_DICT_LEXICON
import re
import enchant
d = enchant.Dict("en_US")
# g2p = G2p()

with open('./tts_front/lexicon26.txt', 'r', encoding='utf-8') as fo:
    global GLOBAL_ENGLISH_DICT_LEXICON
    GLOBAL_ENGLISH_DICT_LEXICON = dict()
    lines = fo.readlines()
    for line in lines:
        character, phone, character_pinyin = line.split('\t')
        GLOBAL_ENGLISH_DICT_LEXICON[character] = phone

def ch2py(chinese:str,u2v=True):
    pinyin = chinese_bian_diao_pinyin(chinese)
    if u2v:
        pinyin = split_phoneme.u_to_v(pinyin)

    return pinyin
def words_flag(string:str):
    if string.isupper():
        flag_now = 0
    elif string.islower():
        flag_now = 1
    else:
        flag_now = 2
    return flag_now
def en_connect(connect_before:list):
    count_en=0
    for cb in connect_before:
        if cb.isupper():
            count_en=count_en+1
    connect_after=''
    for i in range(0,len(connect_before)):
        if count_en>1:
            if connect_before[i].isupper():
                connect_after=connect_after+ connect_before[i]+' / '
                count_en=count_en-1
            else:
                connect_after = connect_after + connect_before[i]+' '
        else:
            if i<len(connect_before)-1:
                connect_after = connect_after + connect_before[i]+' '
            else:
                connect_after = connect_after + connect_before[i]
    return connect_after

def split_chinese_eng(string:str):
    '''
    将英文和非英文分开，例如string='hello hello你好。'中英文分开后为list=['hello',' ','hello','你好']

    :param string: 中英文分开前，字符串
    :return: 中英文分开后，list
    '''
    if string == '':
        return ''
    flag=-1
    list_final=[]
    str_word = ''
    flag_begin=0
    ###中英文分开
    for ph in string:
        flag_begin=flag_begin+1
        if ph.isupper() or ph.islower():
            flag_now=0
        else:
            flag_now=1
        if flag_now==flag:
            str_word=str_word+ph
        else:
            if flag_begin ==1:
                str_word = str_word + ph
                flag = flag_now
            else:
                list_final.append(str_word)
                flag=flag_now
                str_word=ph
    list_final.append(str_word)
    return list_final
def firt_word_flag_after_a(i,list_final):
    '''
    检查a后第一个单词或者汉字，若为单词则为False按照AH0读，若为汉字或者字符则为True按照EA1读
    :param i: a所在的索引
    :param list_final: 中英文分割后的列表
    :return:按照单词读为False，按照字母读为True
    '''
    zhmodel = re.compile(u'[\u4e00-\u9fa5]')
    i = i + 1
    while i < len(list_final):
        if list_final[i].isupper() or list_final[i].islower():
            if len(list_final[i])>1 and (d.check(list_final[i]) or d.check(list_final[i].title())):
                return False
            else:
                return True
        else:
            match = zhmodel.search(list_final[i])
            if match:
                return True
        i = i + 1
    return True

def trans2phone(list_final:list,chinese_split=True,chinese_u2v=True):
    '''
    将单独的英文和中文转化为音素
    :param list_final:中英文分割后的列表
    :return:中英文分割后的列表转化为音素
    '''
    phone_final = []
    zimu_read=['a','it','led','mm','id']
    ###中文转化为拼音，英文转化为音素
    for i,word in enumerate(list_final):
        if word[0].islower() or word[0].isupper():
            # if word in words.words():
            flag_check = d.check(word) or d.check(word.title())  ###True为单词，False为字母
            # flag_check = False  #仅仅处理为字母，全部按照字母读
            if (word.lower() in zimu_read)and firt_word_flag_after_a(i, list_final):
                flag_check=False
            if flag_check:
                # en_phone = g2p(word)
                en_phone = pronouncing.phones_for_word(word)
                # phone_tem = ' '.join(en_phone)
                if len(en_phone) == 0 or (not en_phone[0].isupper()):
                    word = word.upper()
                    phone_tem_zimu = []
                    for w in word:
                        en_phone = GLOBAL_ENGLISH_DICT_LEXICON[w]
                        en_phone = ''.join(en_phone)
                        phone_tem_zimu.append(en_phone)
                    phone_tem = ' / '.join(phone_tem_zimu)
                else:
                    phone_tem = en_phone[0]

            else:
                word = word.lower()
                # if word in words.words():###判断是否为英文单词
                flag_check = d.check(word)  ###True为单词，False为字母
                # flag_check=False##全部按照字母读
                if (word.lower() in zimu_read ) and firt_word_flag_after_a(i, list_final):
                    flag_check = False
                if flag_check:
                    # en_phone = g2p(word)
                    en_phone = pronouncing.phones_for_word(word)
                    if len(en_phone) == 0 or (not en_phone[0].isupper()):
                        word = word.upper()
                        phone_tem_zimu = []
                        for w in word:
                            en_phone = GLOBAL_ENGLISH_DICT_LEXICON[w]
                            en_phone = ''.join(en_phone)
                            phone_tem_zimu.append(en_phone)
                        phone_tem = ' / '.join(phone_tem_zimu)
                    else:
                        phone_tem = en_phone[0]
                    # phone_tem = ' '.join(en_phone)
                else:
                    #####字母处理
                    word = word.upper()
                    phone_tem_zimu = []
                    for w in word:
                        en_phone = GLOBAL_ENGLISH_DICT_LEXICON[w]
                        en_phone = ''.join(en_phone)
                        phone_tem_zimu.append(en_phone)
                    phone_tem = ' / '.join(phone_tem_zimu)
        else:
            if word == ' ':
                continue
            else:
                pattern = '#1|#2|#3|#4'
                pattern1 = '#'
                string_lists_with_rhy_num=re.split(pattern1, word)
                no_rhy_chinese=re.sub(pattern=pattern,repl='',string=word)
                string_phone = ch2py(no_rhy_chinese, u2v=chinese_u2v)
                split_string_phone = string_phone.split()
                string_lists = re.split(pattern, word)
                # phone_list = []
                phone_tem = ''
                index_now = 0
                first_flag = 0
                for i in range(len(string_lists)):
                    # if first_flag:
                    #     rhy_num = string_list[0]
                    #     string_list = string_list[1:]
                    if first_flag:
                        rhy_num = string_lists_with_rhy_num[i][0]
                        string_list = string_lists_with_rhy_num[i][1:]
                    else:
                        string_list = string_lists_with_rhy_num[i]
                    if '儿' in string_list:
                        string_phone_temp = ch2py(string_list,u2v=False)
                        string_phone = ' '.join(split_string_phone[index_now:index_now + len(string_phone_temp.split())])
                        index_now = index_now + len(string_phone_temp.split())
                    else:
                        string_phone = ' '.join(split_string_phone[index_now:index_now+len(string_list)])
                        index_now = index_now+len(string_list)

                    if chinese_split:
                        string_phone = split_phoneme.split_sheng(string_phone)
                    if first_flag:
                        # phone_list.append(' #'+rhy_num+' '+string_phone)
                        phone_tem =phone_tem + ' #'+rhy_num+' '+string_phone
                    else:
                        phone_tem = phone_tem +string_phone
                        # phone_list.append(string_phone)

                    first_flag = 1
                # phone_tem = ' #1 '.join(phone_list)
        phone_final.append(phone_tem)
    return phone_final
def add_ch_en_segment_sgin(phone_final):
    flag_before = words_flag(phone_final[0])
    list_temp = []
    list_final_ph = []
    flag = 1
    for i in range(0, len(phone_final)):
        if flag:
            if words_flag(phone_final[i]) == 0:
                flag_before = 0
                flag = 0
                list_temp.append(phone_final[i])

            elif words_flag(phone_final[i]) == 1:
                flag_before = 1
                flag = 0
                list_temp.append(phone_final[i])

            else:
                list_temp.append(phone_final[i])

        else:
            flag_now = words_flag(phone_final[i])
            if flag_now == 2:
                flag_now = flag_before
            if flag_now == flag_before:
                list_temp.append(phone_final[i])
            else:
                if flag_before == 1:
                    str_temp = ''.join(list_temp)
                    list_temp = str_temp
                else:
                    list_temp = en_connect(list_temp)
                list_final_ph.append(list_temp)
                list_temp = []
                flag_before = words_flag(phone_final[i])
                list_temp.append(phone_final[i])
        if i == len(phone_final) - 1:
            if flag_before == 1:
                str_temp = ''.join(list_temp)
                list_temp = str_temp
            else:
                list_temp = en_connect(list_temp)
            list_final_ph.append(list_temp)

    phone_str = ' / '.join(list_final_ph)
    phone_str = phone_str.replace('，', ',').replace('。', '.')#.replace('  ', ' ')
    if '#' in phone_str:
        phone_list = phone_str.split('#')
        phone_str = '#'.join(phone_list[:-1])
        phone_str = phone_str + '#2' + phone_list[-1][1:]
    # else:
    #     phone_str=phone_list[0]
    return phone_str
def word2phone(string,chinese_split=True,chinese_u2v=True):
    if string=='':
        return ''
    split_chinese_eng_list=split_chinese_eng(string)
    phone_split_list = trans2phone(split_chinese_eng_list,chinese_split=chinese_split,chinese_u2v=chinese_u2v)
    phone_str = add_ch_en_segment_sgin(phone_split_list)
    # phone_str = phone_str.replace('  ',' ')
    return phone_str
def word2phone1(string):
    if string == '':
        return ''
    flag=-1
    list_final=[]
    str_word = ''
    flag_begin=0
    ###中英文分开
    for ph in string:
        flag_begin=flag_begin+1
        if ph.isupper() or ph.islower():
            flag_now=0
        else:
            flag_now=1
        if flag_now==flag:
            str_word=str_word+ph
        else:
            if flag_begin ==1:
                str_word = str_word + ph
                flag = flag_now
            else:
                list_final.append(str_word)
                flag=flag_now
                str_word=ph
    list_final.append(str_word)
    # print(list_final)
    phone_final = []
    ###中文转化为拼音，英文转化为音素
    for word in list_final:

        if word[0].islower() or word[0].isupper():
            # if word in words.words():
            flag_check = d.check(word) or d.check(word.title())###True为单词，False为字母
            # flag_check = False  #仅仅处理为字母，全部按照字母读
            if flag_check:
                # en_phone = g2p(word)
                en_phone = pronouncing.phones_for_word(word)
                # phone_tem = ' '.join(en_phone)
                if len(en_phone) == 0 or (not en_phone[0].isupper()):
                    word = word.upper()
                    phone_tem_zimu = []
                    for w in word:
                        en_phone = GLOBAL_ENGLISH_DICT_LEXICON[w]
                        en_phone = ''.join(en_phone)
                        phone_tem_zimu.append(en_phone)
                    phone_tem = ' / '.join(phone_tem_zimu)
                else:
                    phone_tem = en_phone[0]

            else:
                word = word.lower()
                # if word in words.words():###判断是否为英文单词
                flag_check = d.check(word)###True为单词，False为字母
                # flag_check=False##全部按照字母读
                if flag_check:
                    # en_phone = g2p(word)
                    en_phone = pronouncing.phones_for_word(word)
                    if len(en_phone)==0 or (not en_phone[0].isupper()):
                        word = word.upper()
                        phone_tem_zimu = []
                        for w in word:
                            en_phone = GLOBAL_ENGLISH_DICT_LEXICON[w]
                            en_phone = ''.join(en_phone)
                            phone_tem_zimu.append(en_phone)
                        phone_tem = ' / '.join(phone_tem_zimu)
                    else:
                        phone_tem = en_phone[0]
                    # phone_tem = ' '.join(en_phone)
                else:
                    #####字母处理
                    word = word.upper()
                    phone_tem_zimu = []
                    for w in word:
                        en_phone = GLOBAL_ENGLISH_DICT_LEXICON[w]
                        en_phone = ''.join(en_phone)
                        phone_tem_zimu.append(en_phone)
                    phone_tem = ' / '.join(phone_tem_zimu)
        else:
            if word==' ':
                continue
            else:
                # pattern = '#1|#2|#3|#4'
                pattern='#'
                string_lists = re.split(pattern, word)
                # phone_list = []
                phone_tem = ''
                firt_flag = 1
                for string_list in string_lists:
                    if firt_flag:
                        string_phone = ch2py(string_list,split=False)
                        firt_flag=0
                        phone_tem = phone_tem+string_phone
                    else:
                    # phone_list.append(string_phone)
                        string_phone = ch2py(string_list[1:], split=False)
                        phone_tem = phone_tem + ' #'+string_list[0]+' '+string_phone
                # phone_tem = ' #1 '.join(phone_list)
        phone_final.append(phone_tem)
    # print(phone_final)
    #####添加中英文分割符号
    flag_before = words_flag(phone_final[0])
    list_temp=[]
    list_final_ph = []
    flag=1
    for i in range(0,len(phone_final)):
        if flag:
            if words_flag(phone_final[i])==0:
                flag_before=0
                flag = 0
                list_temp.append(phone_final[i])

            elif words_flag(phone_final[i])==1:
                flag_before = 1
                flag = 0
                list_temp.append(phone_final[i])

            else:
                list_temp.append(phone_final[i])

        else:
            flag_now=words_flag(phone_final[i])
            if flag_now==2:
                flag_now=flag_before
            if flag_now==flag_before:
                list_temp.append(phone_final[i])
            else:
                if flag_before==1:
                    str_temp = ''.join(list_temp)
                    list_temp=str_temp
                else:
                    list_temp = en_connect(list_temp)
                list_final_ph.append(list_temp)
                list_temp=[]
                flag_before=words_flag(phone_final[i])
                list_temp.append(phone_final[i])
        if i ==len(phone_final)-1:
            if flag_before==1:
                str_temp = ''.join(list_temp)
                list_temp=str_temp
            else:
                list_temp = en_connect(list_temp)
            list_final_ph.append(list_temp)
    # print(list_final_ph)

    phone_str = ' / '.join(list_final_ph)
    phone_str = phone_str.replace('，',',').replace('。','.')#.replace('  ',' ')
    if '#' in phone_str:
        phone_list = phone_str.split('#')
        phone_str='#'.join(phone_list[:-1])
        # print(phone_list)
        phone_str = phone_str+'#2'+phone_list[-1][1:]
    # else:
    #     phone_str=phone_list[0]
    return phone_str

if __name__=='__main__':
    string = "To say a few words on the principles of design in typography ."
    # string = string.replace('#2','').replace('#1','').replace('#3','').replace('#4','')
    print(word2phone(string))

    # print('bx' in words.words())
    # print(g2p('bx'))