# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
import tts_front.ChineseRhythmPredictor.models.synthesize as syn
import re


class ModelRhythm:
    def __init__(self, sentence_len=14, long_sentence_add=True, min_rhy_len=5):
        self.sentence_len = sentence_len
        self.long_sentence_add = long_sentence_add
        self.min_rhy_len = min_rhy_len
        self.rhyth_model = syn.Synthesizer()
        self.rhyth_model.load("./tts_front/ChineseRhythmPredictor/models/models/test/bilstm/my-model-5.meta",
                              "./tts_front/ChineseRhythmPredictor/models/models/test/bilstm/my-model-5")

    def long_sentence_add_rhy(self, chinese_rm_symbols_list):
        """
        长句子添加韵律，句子字符长度大于sentence_len时，用模型预测韵律
        :param chinese_rm_symbols_list:断句后的原始列表 self.sentence_len:触发韵律模型的长度，当句子大于sentence_len时，用韵律模型断句；小于时用在句尾添加韵律
        :return:字符串，添加韵律后的句子
        """
        chinese_add_rhy_final = ''
        for chinese_rm_symbol in chinese_rm_symbols_list:
            if len(chinese_rm_symbol) > self.sentence_len:
                try:
                    chinese_rhy = self.add_rhy(chinese_rm_symbol)
                    if '#' not in chinese_rhy:  # #若模型重没有韵律符号#，则在句尾添加韵律符号。
                        chinese_rhy = ending_add_rhy(chinese_rhy)
                    chinese_rhy = self.merge_rh(chinese_rhy)  # 4
                except Exception as e:
                    print('long sentence add rhy model add error', e.args)
                    chinese_rhy = ending_add_rhy(chinese_rm_symbol)
            else:
                chinese_rhy = ending_add_rhy(chinese_rm_symbol)
            chinese_add_rhy_final = chinese_add_rhy_final + chinese_rhy
        return chinese_add_rhy_final

    def add_rhy(self, chinese: str):  # ##模型添加韵律
        """
        模型LSTM模型预测韵律
        基本思路：检查是否有中文，含有中文，则用韵律模型预测，若无中文则用结尾添加。韵律预测后若无韵律符号，则在句尾添加韵律
        :param chinese: 待预测的汉语
        :return: 加韵律的句子str
        """
        zhmodel = re.compile(u'[\u4e00-\u9fa5]')  # 检查是否含有中文
        match = zhmodel.search(chinese)
        if match:
            chinese_with_rhy = self.rhyth_model.pred(chinese)  # 空格一进模型就变没了，所以空格得用*号进行占位
            chinese_rhy = chinese_with_rhy.replace("#1", '').replace('#2', '#1')
            chinese_rhy = chinese_rhy.split('#')
            if len(chinese_rhy) > 1:
                chinese_rhy_str = '#'.join(chinese_rhy[:-1])
                chinese_end = chinese_rhy[-1].strip()
                if ',' in chinese_end[1:] or '。' in chinese_end[1:]:
                    # ##判断最后一个#后是否含有汉字
                    if zhmodel.search(chinese_end) or chinese_end.lower().islower():
                        chinese_rhy_final = chinese_rhy_str + '#' + chinese_end[:-1] + '#2' + chinese_end[-1]
                    else:
                        chinese_rhy_final = chinese_rhy_str + '#2' + chinese_end[1:]
                else:
                    if len(chinese_end) > 1:
                        chinese_rhy_final = chinese_rhy_str + '#' + chinese_end + '#2.'
                    else:
                        chinese_rhy_final = chinese_rhy_str + '#2' + chinese_end[1:]

            else:
                chinese_rhy_str = chinese_rhy[0].strip()
                chinese_rhy_final = chinese_rhy_str.replace('。', '#2.').replace('，', '#2,').replace('#1#2', '#2')
        else:
            chinese_rhy_final = ending_add_rhy(chinese)  # 纯英文添加韵律
        chinese_rhy_final = chinese_rhy_final.replace('#1,#2', '#2')
        return chinese_rhy_final

    def merge_rh(self, chinese):
        """
        将#1的短词进行合并，合并规则是长度小于min_rhy_len个字的词，将与前后字数较少的组合并
        :param chinese: 韵律模型生成的长句子韵律
        :return: 合并后的韵律句子
        """
        chinese_split = chinese.split('#1')
        chinese_split[-1], end_chinese = chinese_split[-1].split('#')
        new = [chinese_split[0]]  # #最终的句子
        flag = 1 if len(chinese_split[0]) < self.min_rhy_len else 0
        for i in range(1, len(chinese_split)):
            if len(chinese_split[i]) < self.min_rhy_len:
                if i < len(chinese_split) - 1:
                    if len(new[-1]) > len(chinese_split[i + 1]):
                        if flag:
                            new[-1] = new[-1] + chinese_split[i]
                            flag = 0
                        else:
                            if len(new[-1]) < self.min_rhy_len:
                                new[-1] = new[-1] + chinese_split[i]
                                flag = 0
                            else:
                                new.append(chinese_split[i])
                                flag = 1
                    else:
                        new[-1] = new[-1] + chinese_split[i]
                        flag = 0
                else:
                    new[-1] = new[-1] + chinese_split[i]
            else:
                if flag:
                    new[-1] = new[-1] + chinese_split[i]
                    flag = 0
                else:
                    if len(new[-1]) < self.min_rhy_len:
                        new[-1] = new[-1] + chinese_split[i]
                    else:
                        new.append(chinese_split[i])
        return '#1'.join(new) + '#' + end_chinese

    def model_rhy(self, chinese_rm_symbols_list, chinese_rm_symbols):
        """
        模型预测入口
        :param chinese_rm_symbols_list: 中文列表，例如：['行政部将定期开展IT,', '数码设备品牌服务日活动。']
        :param chinese_rm_symbols: 中文字符串， 例如：行政部将定期开展IT,数码设备品牌服务日活动
        :return:
        """
        if self.long_sentence_add:
            chinese_rhy = self.long_sentence_add_rhy(chinese_rm_symbols_list)
        else:
            try:
                chinese_rhy = self.add_rhy(chinese_rm_symbols)
                if '#' not in chinese_rhy:
                    chinese_rhy = ending_add_rhy(chinese_rhy)
            except Exception as e:
                print('add error ', e.args)
                chinese_rhy = ending_add_rhy(chinese_rm_symbols)
        return chinese_rhy


def ending_add_rhy(chinese: str):
    """
    在汉语句末加韵律。
    :param chinese:汉语句子，字符串
    :return: 句末加韵律的汉语
    """
    chinese = chinese.replace(',', '#2,').replace('!', '#2!').replace('?', '#2?') \
        .replace('，', '#2,').replace('。', '#2.').replace('！', '#2!').replace('？', '#2?')  # 将所有的标点符号前加上#2，
    if len(chinese) > 2 and chinese[-3] == '#':
        chinese = chinese[:-2] + '2' + chinese[-1]
    else:
        chinese = chinese + '#2.'
    return chinese
