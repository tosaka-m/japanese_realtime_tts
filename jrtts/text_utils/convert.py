#coding:utf-8
from functools import reduce
import jaconv
import re
import MeCab
import unicodedata
import zenhan
from pykakasi import kakasi
from .yomi2voca import mapper_no_space_removeST, keta_mapper
import logging
logger = logging.getLogger(__name__)
mecab = MeCab.Tagger('-Ochasen')
mecab_yomi = MeCab.Tagger('-Oyomi')
digit_mapper = {i: mecab_yomi.parse(zenhan.h2z(str(i)))[:-1] for i in range(10)}
_kakasi = kakasi()
_kakasi.setMode('J', 'H')
kakasi_converter = _kakasi.getConverter()


def hiragana2onso(text):
    orig = text

    # 空白除去パターン
    text = _remove_special_space(text)

    for k, v in mapper_no_space_removeST.items():
        text = text.replace(k, v)

    text= _kigou2roman(text, add_space=False)
    return text

def text2hiragana(text):
    text = unicodedata.normalize("NFKC", text)
    text = normalize_neologd(text)
    text = text.replace(' ', '　')

    parsed = mecab.parse(text).split('\n')
    parsed = [p.split('\t') for p in parsed]
    way_of_readings = [_special_char_convert(p[1], p[3], idx==0)
                       for idx, p in enumerate(parsed) if len(p) >= 2]
    way_of_reading = "".join(way_of_readings)

    way_of_reading = _num2word(way_of_reading)
    way_of_reading = kakasi_converter.do(way_of_reading)
    way_of_reading = jaconv.kata2hira(way_of_reading)
    way_of_reading = way_of_reading.replace('　', ' ')
    way_of_reading = re.sub(r'[^ぁ-ゔ。、！？ー\.\!\?,\s]', '', way_of_reading)
    way_of_reading = re.sub(r'\s{2,}', ' ', way_of_reading)
    way_of_reading = re.sub(r'^\s', '', way_of_reading)

    if len(way_of_reading) == 0:
        way_of_reading = '.'
    if way_of_reading[-1] != '.':
        way_of_reading = way_of_reading + '.'
    if way_of_reading[0] != ' ':
        way_of_reading = ' ' + way_of_reading

    return way_of_reading

def align_text_and_f0_convert(text, f0):
    hiragana = text2hiragana(text)
    _hiragana = _remove_special_space(hiragana)
    for i, (k, v) in enumerate(mapper_no_space_removeST.items()):
        _hiragana = _hiragana.replace(k, '[%d]' % len(v))
    _process_hiragana = _kigou2roman(_hiragana)
    _process_hiragana = re.sub(r'[^\d\[\]]', '[0]', _process_hiragana)
    _process_hiragana = re.findall(r'\[\d\]', _process_hiragana)

    #hiragana_flag = [(not (re.match(r'\d', h) is None)) for h in _hiragana]
    hiragana_flag = [(h[1] != '0') for h in _process_hiragana]
    hiragana_index = [i for i, h in enumerate(_process_hiragana) if hiragana_flag[i]]
    #print(hiragana)
    f0_mapper = {j: [f0[i]] * int(_process_hiragana[j][1]) for i, j in enumerate(hiragana_index)}
    f0 = [f0_mapper[i] if flag else [0] for i, flag in enumerate(hiragana_flag)]
    #f0 = [re.sub(r'[^\d]', '0', _kigou2roman(value)) for value in f0]
    print(f0)
    f0 = list(reduce(lambda x,y: x+y, f0))
    onso = hiragana2onso(hiragana)
    assert(len(onso)==len(f0))
    return onso, f0

def text2splithiragana(text):
    hiragana = text2hiragana(text)
    _hiragana = _remove_special_space(hiragana)
    mapper = {}
    for i, (k, v) in enumerate(mapper_no_space_removeST.items()):
        replace_value = '[%d]' % i
        _hiragana = _hiragana.replace(k, replace_value)
        mapper[replace_value] = k
    split_hiragana_index = re.findall(r'\[\d+\]', _hiragana)
    split_hiragana = list(map(lambda x: mapper[x], split_hiragana_index))
    return split_hiragana


# ------------------------------------------------------------ #
# --- Tips --------------------------------------------------- #
# ------------------------------------------------------------ #

def _glowtts_replacing(onso):
    onso = onso.replace(',', ' ')
    onso = onso.replace(';', ',')
    return onso

def _kigou2roman(text, add_space=False):
    space = ' ' if add_space else ''
    text = re.sub('。', space + '.', text)
    text = re.sub('、', space + ',', text)
    text = re.sub('\.\.\.', space + ';', text)

    if add_space:
        text = re.sub(r'\!', ' !', text)
        text = re.sub(r'\?', ' ?', text)

    # 空白除去パターン
    for pattern in ['!', '?', '.', ',', ';']:
        text = text.replace(" %s" % pattern ,pattern)

    text = _glowtts_replacing(text)
    return text

def _num2word(text):
    nums = re.findall('\d+', text)
    if len(nums) == 0:
        return text
    nums_map = {num:__num2word(num) for num in nums}
    for k, v in nums_map.items():
        text = re.sub(k, v, text)
    return text

def __num2word(num):
    textnum = str(num)
    nchars = len(textnum)
    chars = []
    start_zero = (textnum[0] == '0')
    for i in range(nchars):
        try:
            ketayomi = keta_mapper[nchars - i]
        except KeyError:
            ketayomi = ''
        num = textnum[i]
        yomi = digit_mapper[int(num)]

        if (not start_zero) and (num == '0'):
            yomi = ''
            ketayomi = ''

        if not ((nchars - i - 1) >= 1 and num == '1'):
            chars.append(yomi)
        if (not start_zero):
            chars.append(ketayomi)
    word = ''.join(chars)
    return word

def _special_char_convert(text, texttype, init=False):
    if init:
        return text
    if not ('助詞' in texttype):
        return text
    else:
        if text == 'ハ':
            text = 'ワ'
        elif text == 'ヲ':
            text = 'オ'
        elif text == 'ヘ':
            text = 'エ'
    return text

def _remove_special_space(text):
    for pattern in ['ゃ', 'ゅ', 'ょ', 'っ']:
        text = text.replace(" %s" % pattern ,pattern)
    return text

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s


def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    return s
    # # --- not remove space ---
    # blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
    #                   '\u3040-\u309F',  # HIRAGANA
    #                   '\u30A0-\u30FF',  # KATAKANA
    #                   '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
    #                   '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
    #                   ))
    # basic_latin = '\u0000-\u007F'

    # def remove_space_between(cls1, cls2, s):
    #     p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
    #     while p.search(s):
    #         s = p.sub(r'\1\2', s)
    #     return s

    # s = remove_space_between(blocks, blocks, s)
    # s = remove_space_between(blocks, basic_latin, s)
    # s = remove_space_between(basic_latin, blocks, s)
    # return s


def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                  '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s


if __name__=='__main__':
    sample = 'あと鶏肉ハンバーグは どうする ?'
    hiragana = text2hiragana(sample)
    onso = hiragana2onso(hiragana)
    print(hiragana, onso)



