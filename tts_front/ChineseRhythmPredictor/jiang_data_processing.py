# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
from gensim.models import word2vec
from tts_front.ChineseRhythmPredictor.parameter import MAX_SENTENCE_SIZE
# from parameter import WORD_EMBEDDING_SIZE
# from parameter import CHAR_EMBEDDING_SIZE
import numpy as np
import pandas as pd
import re
import jieba.posseg as posseg

# def read_component():
#     #读取words和ids的dataframe
#     df_words_ids=pd.read_csv(filepath_or_buffer="./tts_front/ChineseRhythmPredictor/data/dataset/words_ids.csv",encoding="utf-8")
#     #读取tags和ids的dataframe
#     df_tags_ids=pd.read_csv(filepath_or_buffer="./tts_front/ChineseRhythmPredictor/data/dataset/tags_ids.csv",encoding="utf-8")
#     df_pos_ids = pd.read_csv(filepath_or_buffer="./tts_front/ChineseRhythmPredictor/data/dataset/pos_ids.csv",encoding="utf-8")
#
#     #转换为words2id, id2words, tags2id, id2tags
#     #df_data=pd.DataFrame(data={})
#     words2id=pd.Series(data=df_words_ids["id"].values,index=df_words_ids["words"].values)
#     id2words=pd.Series(data=df_words_ids["words"].values,index=df_words_ids["id"].values)
#     tags2id = pd.Series(data=df_tags_ids["id"].values, index=df_tags_ids["tags"].values)
#     id2tags = pd.Series(data=df_tags_ids["tags"].values, index=df_tags_ids["id"].values)
#     pos2ids = pd.Series(data=df_pos_ids["id"].values,index=df_pos_ids["pos"].values)
#     return words2id, id2words, tags2id, id2tags,pos2ids

df_words_ids = pd.read_csv(filepath_or_buffer="./tts_front/ChineseRhythmPredictor/data/dataset/words_ids.csv",
                           encoding="utf-8")
# 读取tags和ids的dataframe
df_tags_ids = pd.read_csv(filepath_or_buffer="./tts_front/ChineseRhythmPredictor/data/dataset/tags_ids.csv",
                          encoding="utf-8")
df_pos_ids = pd.read_csv(filepath_or_buffer="./tts_front/ChineseRhythmPredictor/data/dataset/pos_ids.csv",
                         encoding="utf-8")

# 转换为words2id, id2words, tags2id, id2tags
# df_data=pd.DataFrame(data={})
words2id = pd.Series(data=df_words_ids["id"].values, index=df_words_ids["words"].values)
id2words = pd.Series(data=df_words_ids["words"].values, index=df_words_ids["id"].values)
tags2id = pd.Series(data=df_tags_ids["id"].values, index=df_tags_ids["tags"].values)
id2tags = pd.Series(data=df_tags_ids["tags"].values, index=df_tags_ids["id"].values)
pos2ids = pd.Series(data=df_pos_ids["id"].values, index=df_pos_ids["pos"].values)


def padding(ids):
    # ids = list(words2id[sentence])
    ids = list(ids)
    if len(ids) > MAX_SENTENCE_SIZE:  # 超过就截断
        return ids[:MAX_SENTENCE_SIZE]
    if len(ids) < MAX_SENTENCE_SIZE:  # 短了就补齐
        ids.extend([0] * (MAX_SENTENCE_SIZE - len(ids)))
    return ids


# def posids_padding(sentence):
#     pos_ids = list(pos2ids[sentence])
#     if len(pos_ids) > MAX_SENTENCE_SIZE:  # 超过就截断
#         return pos_ids[:MAX_SENTENCE_SIZE]
#     if len(pos_ids) < MAX_SENTENCE_SIZE:  # 短了就补齐
#         pos_ids.extend([0] * (MAX_SENTENCE_SIZE - len(pos_ids)))
#     return pos_ids

def get_pos(chinese):
    pairs = [tuple(pair) for pair in posseg.cut(chinese)]
    s = [pair[0] for pair in pairs]
    pos = [pair[1] for pair in pairs]
    pos_list = []
    length_list = []
    for i in range(0, len(s)):
        for j in range(0, len(s[i])):
            if j == len(s[i]) - 1:
                pos_list.append(pos[i])
                length_list.append(len(s[i]))
            else:
                pos_list.append(0)
                length_list.append(0)
    return np.array(pos_list), np.array(length_list)


def data_main(ch):
    pos_list, length_list = get_pos(ch)
    pattern2 = re.compile(r"[^\s]")
    string = " ".join(re.findall(pattern=pattern2, string=ch))
    ch_list = string.split(' ')
    chinese = np.array(ch_list)
    # words2id, id2words, tags2id, id2tags, pos2ids = read_component()
    X_ids = words2id[chinese]
    # X_test = pd.DataFrame(chinese).apply(X_padding).tolist()#ndarray
    X_test = padding(X_ids)
    len_test = np.array([len(ch_list)])
    # pos_test = pd.DataFrame(pos_list).apply(posids_padding).tolist()
    pos_ids = pos2ids[pos_list]
    pos_test = padding(pos_ids)
    length_test = padding(length_list)
    position_ids = list(range(1, len(ch_list) + 1))
    position_test = padding(position_ids)
    # print(pos_test)
    # print(X_test)
    # # print(len_test)
    # print(length_test)
    # print(position_test)
    return np.array([X_test]), np.array([pos_test]), np.array(len_test), np.array([length_test]), np.array(
        [position_test])


if __name__ == "__main__":
    ch = '你好，欢迎来到汽车之家。'
    X_test, pos_test, len_test, length_test, position_test = data_main(ch)
    print(X_test, pos_test, len_test, length_test, position_test)
#
# chinese.
