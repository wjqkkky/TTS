# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
from tts_front.chinesetone2pinyin import *
import tts_front.remove_special_symbols as rm
from tts_front.ChineseRhythmPredictor import *
from tts_front.ChineseRhythmPredictor.models.bilstm_cbow_pred_jiang_test_haitian import BiLSTM#* as cb #pred_rhy,load_jiang_test
from tts_front.en2phoneme import word2phone
import re
import time

def split_text(text):
    # TODO 切割后的text不包含！？
    '''
    分割符号包括，[,。；\. ? ! -- \n - --]
    :param text:
    :return:
    '''
    #，|：|。|；|？|！|——
    text = text.replace('，',',').replace('？','?').replace('！','!').replace('；',';')
    # pattern_str = "，|。|；|,|？|！|——|\n|-"
    pattern_str = ",|。|;|\?|!|——|\n|-"
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
            if text!='':
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


def regula_specail(chinese: str):
    '''
    匹配特殊符号
    :param chinese:汉字字符串
    :return: 正则化后的字符串
    '''
    ###正则化匹配网址
    # regular = re.compile(r'[a-zA-z]+://[^\s]*')
    regular = re.compile(r'([hH][tT]{2}[pP]://|[hH][tT]{2}[pP][sS]://|[wW]{3}.|[wW][aA][pP].|[fF][tT][pP].|[fF][iI][lL][eE].)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')
    # print(re.findall(regular, chinese))
    chinese = re.sub(pattern=regular, repl="", string=chinese)
    # print(ss)
    ####将'] '、’) ‘,’） ‘，反括号空格替换为逗号，
    pattern = '\) |\] |\} |\） '
    chinese = re.sub(pattern=pattern, repl=",", string=chinese)

    ###正则化匹配(数字)、(1).规则：将反括号前为数字的情况，全部替换为数字加顿号。例如：1）换为1,，（十一）换成十一、
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
            chinese = chinese.replace(parenthese[-1] + ')', parenthese[-1] + ',').replace('(', '')
    chinese = chinese.replace('(', '').replace(')', '')
    # print(re.findall(regular_in,chinese))
    # print(chinese)
    ####正则化匹配“第*条|章”规则，匹配规则：只要为"第***+空格"替换为"第***,"***字符串的个数为3
    # regular = r'[第]+(.*?)+[章 ]'
    # regular = r'([第]+(.*?)+[ ])'
    regular = r'([第](.*?)[ ]+\b)'
    parentheses_list = re.findall(regular, chinese)

    for parenthese in parentheses_list:
        if parenthese[0] != '' and len(parenthese[0]) < 6:
            chinese = chinese.replace(parenthese[0], parenthese[0][:-1] + ',')
    return chinese

def add_rhy(chinese:str,model):
    '''
    模型LSTM模型预测韵律
    :param chinese: 待预测的汉语
    :param model: LSTM模型
    :return: 加韵律的模型
    '''
    # rhy_ch = model.pred_rhy(ch)
    # print(chinese)
    zhmodel = re.compile(u'[\u4e00-\u9fa5]')  # 检查是否含有中文
    match = zhmodel.search(chinese)
    if match:
        chinese_with_rhy = model.pred_rhy(chinese)
        chinese_rhy = chinese_with_rhy.replace("#1", '').replace('#2', '#1')
        chinese_rhy = chinese_rhy.split('#')
        if len(chinese_rhy) > 1:
            chinese_rhy_str = '#'.join(chinese_rhy[:-1])
            if ',' in chinese_rhy[-1][1:] or '。' in chinese_rhy[-1][1:]:
                chinese_rhy_final = chinese_rhy_str + '#2' + chinese_rhy[-1][1:]
            else:
                chinese_end = chinese_rhy[-1].replace('\n','')
                if len(chinese_end)>1:
                    chinese_rhy_final = chinese_rhy_str +'#'+chinese_end+'#2.'
                else:
                    chinese_rhy_final = chinese_rhy_str + '#2' + chinese_rhy[-1][1:]
        else:
            chinese_rhy_str = chinese_rhy[0].replace('\n', '')

            chinese_rhy_final = chinese_rhy_str.replace('。', '#2.').replace('，', '#2,').replace('#1#2','#2')
    else:
        chinese_rhy_final = ending_add_rhy(chinese)#纯英文添加韵律

    return chinese_rhy_final

def ending_add_rhy(chinese:str):
    '''
    在汉语句末加韵律。
    :param chinese:汉语句子，字符串
    :return: 句末加韵律的汉语
    '''
    chinese=chinese.replace(',','#2,').replace('!','#2!').replace('?','#2?')\
        .replace('，','#2,').replace('。','#2.').replace('！','#2!').replace('？','#2?')
    # print(chinese)
    if len(chinese)>2 and chinese[-3] =='#':
        chinese = chinese[:-2]+'2'+chinese[-1]
    else:
        chinese = chinese+'#2.'
    return chinese
def is_continuous_sign(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    string=string.encode('utf-8').decode('utf-8')
    for ch in string:
        if (u'\u4e00' <= ch <= u'\u9fff'):
            return True
        if (ch >= u'\u0041' and ch <= u'\u005a') or (ch >= u'\u0061' and ch <= u'\u007a'):
            return True
        if ch.isdigit():
            return True
    return False

def rm_continuous_sign(chinese):
    '''
    去除连续字符
    :param chinese:去除前的 字符串
    :return: 去除后的字符串
    '''
    pattern = ',|。|，'
    chinese_list = re.split(pattern=pattern, string=chinese)
    # print(chinese_list)
    chinese = ''
    split_chinese_list =[]#返回列表
    flag_begin=1
    for chinese_ in chinese_list:
        if chinese_ == '':
            continue
        if not is_continuous_sign(chinese_):
            continue
        if flag_begin:
            chinese = chinese + chinese_
            flag_begin=0
        else:
            chinese = chinese + ','+chinese_
        split_chinese_list.append(chinese_+',')
    split_chinese_list[-1]=split_chinese_list[-1][:-1]+'。'
    return chinese,split_chinese_list
def merge_rh(chinese,min_rhy_len=4):
    '''
    将#1的短词进行合并，合并规则是长度小于min_rhy_len个字的词，将与前后字数较少的组合并
    :param chinese: 韵律模型生成的长句子韵律
    :return: 合并后的韵律句子
    '''
    chinese_split = chinese.split('#1')
    chinese_split[-1],end_chinese = chinese_split[-1].split('#')
    new = [chinese_split[0]]
    flag = 1 if len(chinese_split[0]) < min_rhy_len else 0
    for i in range(1, len(chinese_split)):
        if len(chinese_split[i]) < min_rhy_len:
            if i < len(chinese_split) - 1:
                if len(new[-1]) > len(chinese_split[i + 1]):
                    if flag:
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
                new.append(chinese_split[i])
    return '#1'.join(new) + '#'+end_chinese

def long_sentence_add_rhy(chinese_rm_symbols_list,model,sentence_len=14):
    '''
    长句子添加韵律，句子字符长度大于sentence_len时，用模型预测韵律
    :param chinese_rm_symbols_list:
    :param model:
    :param sentence_len:
    :return:
    '''
    chinese_add_rhy_final=''
    for chinese_rm_symbol in chinese_rm_symbols_list:
        if len(chinese_rm_symbol)>sentence_len:
            try:
                chinese_rhy = add_rhy(chinese_rm_symbol, model)
                if '#' not in chinese_rhy:
                    # print('ending_add')
                    chinese_rhy = ending_add_rhy(chinese_rhy)
                # print(chinese_rhy)
                chinese_rhy=merge_rh(chinese_rhy,min_rhy_len=4)
                # print(chinese_rhy)
            except:
                # print('ending_add')
                chinese_rhy = ending_add_rhy(chinese_rm_symbol)
        else:
            chinese_rhy = ending_add_rhy(chinese_rm_symbol)
        chinese_add_rhy_final = chinese_add_rhy_final+chinese_rhy
    return chinese_add_rhy_final
def main(chinese:str,model,Long_sentence_add=True):
    chinese = chinese.replace('（', '(').replace('）', ')').replace('、',',').replace('\n','。').replace('\r','。').replace('\t','。')
    ###匹配url，第*章 ，第* ，（反括号前为数字），反括号前为数字）
    chinese=regula_specail(chinese)
    ####多个连续空格保留一个，若最后一个为空格则去除
    chinese = ' '.join(chinese.split())
    if len(chinese)==0:
        return [','],[',']
    # if chinese[-1] == ' ':
    #     chinese = chinese[:-1]
    ####将空格替换为*
    chinese = chinese.replace(' ', '*')
    # print(chinese)
    chinese_list = split_text(chinese)
    ch_rhy_list = []
    phone_list = []
    for ch in chinese_list:
        star_time = time.time()
        chinese_nor = NSWNormalizer(ch).normalize()
        chinese_nor = chinese_nor.replace(":", ',').replace('.','。')
        end_nor_time = time.time()
        chinese_rm_symbols = rm.remove_symbols(chinese_nor)
        chinese_rm_symbols,chinese_rm_symbols_list = rm_continuous_sign(chinese_rm_symbols)#chinese_rm_symbols为str，chinese_rm_symbols_list为列表
        end_rm_time = time.time()

        if model =='end':
            chinese_rhy = ending_add_rhy(chinese_rm_symbols)
            # chinese2phone = word2phone(chinese_rhy)
        elif model =='None':
            chinese_rhy = chinese_rm_symbols


        else:
            if Long_sentence_add:
                chinese_rhy = long_sentence_add_rhy(chinese_rm_symbols_list, model,sentence_len=14)
            else:
                try:
                    chinese_rhy = add_rhy(chinese_rm_symbols, model)
                    if '#' not in chinese_rhy:
                        # print('ending_add')
                        chinese_rhy = ending_add_rhy(chinese_rhy)
                except:
                    # print('ending_add')
                    chinese_rhy = ending_add_rhy(chinese_rm_symbols)

        chinese2phone = word2phone(chinese_rhy,chinese_split=True,chinese_u2v=True)
        end_add_rhy_time = time.time()
        chinese2phone = chinese2phone.replace('*', ' ')
        ####去除空格，多个空格保留一个
        chinese2phone = ' '.join(chinese2phone.split())
        if chinese2phone[-1].isdigit():
            chinese2phone = chinese2phone+' .'
        if model != 'None':
            chinese_rhy = chinese_rhy.replace('\n','')
            chinese_rhy = chinese_rhy[:-2]+'3'+chinese_rhy[-1]
            chinese2phone = chinese2phone[:-3]+'3 .'
            chinese_rhy = chinese_rhy.replace(',','').replace('，','').replace('.','').replace('。','')
            chinese2phone = chinese2phone.replace(' ,','').replace(' .','')

        ch_rhy_list.append(chinese_rhy)
        phone_list.append(chinese2phone)
    return ch_rhy_list,phone_list


if __name__=='__main__':
    model = BiLSTM()
    model.load_model()
    # chinese = '第三条 遵循原则\n\n(一) 合法合规原则:符合《中华人民共和国网络安全法》及国家其它相关法律法规。\n\n(二) 风险可控原则:加强数据风险的事前、事中及事后控制，保持管控力度，持续对外传数据进行识别、监督和分析，确保数据风险可控。\n\n(三) 互惠互利原则:数据资源外传和访问应有具体的业务合作场景，且符合对等、互惠互利原则，要遵守相关法律法规规定。\n\n(四) 合理必要原则:数据外传和访问字段的范围应明确，且与业务合作 场景有关联，遵循"最小授权"、"必须知道"的原则。外传原始数据应根据要求进行脱敏，优先考虑转化为能满足业务合作需要的"标签"、 "画像"或汇总数据的方案。'
    # start_time = time.clock()
    # chinese = chinese.replace('#1', '').replace('#2', '').replace('#3', '').replace('#4', '')
    # ch_rhy_list, phone_list = main(chinese, model)
    # print(ch_rhy_list)
    # print(phone_list)
    # end_time = time.clock()
    # print('total time cost : ',end_time-start_time)
    # # print(pinyin)

    while True:
        chinese=input('输入文字：')
        print(chinese)
        if chinese=='end':
            break
        start_time = time.clock()
        chinese = chinese.replace('#1', '').replace('#2', '').replace('#3', '').replace('#4', '')
        if chinese!='\n':
            ch_rhy_list, phone_list = main(chinese, model)

            print(ch_rhy_list)
            print(phone_list)
            end_time = time.clock()
            print('total time cost : ',end_time-start_time)
        # print(pinyin)

    ##########文本转换
    # file_input = '../haitian_all.txt'
    # file_out = '../haitian_all_yunlv.txt'
    # fo = open(file_out, "w", encoding="utf-8")
    # start_time = time.clock()
    # with open(file_input, "r", encoding="utf-8") as f:
    #     # while 1:
    #     lines = f.readlines()
    #     for i in range(0,len(lines)//2):
    #         # fo.write(line_1)
    #         number,line_1 = lines[2*i].split('\t')
    #         pinyin = lines[2*i+1]
    #         chinese = line_1.replace('#1', '').replace('#2', '').replace('#3', '').replace('#4', '')
    #         ch_rhy_list, phone_list = main(chinese, model)
    #         # pinyin,chinese_normal = ch2py(line_1)
    #         fo.write(line_1)
    #         fo.write(ch_rhy_list[0][0])
    #         fo.write(pinyin)
    #         # print(pinyin)
    #
    #         print(ch_rhy_list)
    #         fo.write(phone_list[0][0])
    #         print(phone_list)
    #         fo.write('\n')
    #         fo.write('\n')
    # # chinese_Normal,pinyin = without_Rhythm(chinese,split=False)
    # # print(chinese_Normal)
    # # print(pinyin)
    # end_time = time.clock()
    # print('total time cost : ',end_time-start_time)
    # f.close()
    # fo.close()
