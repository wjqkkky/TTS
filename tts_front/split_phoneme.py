def split_sheng(words):
    '''
    :param words: 一段未拆分的拼音，拼音和标点和字符之间要用一个空格空起来
    :return: 拆分好的拼音。拆分为声母，韵母。
    '''
    list_sheng = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'j', 'q', 'x', 'zh',
                  'ch', 'sh', 'h', 'r', 'z', 'c', 's', 'w', 'y']
    words = words.replace('  ', ' ')
    words = words.split(' ')
    word_split = []
    for word in words:
        if word == '':
            continue
        if word[0] in list_sheng:
            if word[:2] in list_sheng:
                word_split.append(word[:2])
                word_split.append(word[2:])
                continue
            else:
                word_split.append(word[0])
                word_split.append(word[1:])
        else:
            word_split.append(word)
    str_pinyin = ' '.join(word_split)
    return str_pinyin


def u_to_v(pinyin: str):
    '''
    特殊情况v见了jqxy去了两点还念v,将jqxy后的u改为v
    :param word:汉语拼音
    :return:修订后的汉语拼音
    '''
    list_special = ['y', 'j', 'q', 'x']
    words = pinyin.split()
    final_pinyin = []
    for word in words:
        if word[0] in list_special:
            if word[1] == 'u':
                word = word.replace('u', 'v')
                final_pinyin.append(word)
            else:
                final_pinyin.append(word)
        else:
            final_pinyin.append(word)
    pinyin = ' '.join(final_pinyin)
    return pinyin
