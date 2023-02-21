# -*- coding: utf-8 -*-
"""
#这是包含SS3分类器实现的主模块。

"""
from __future__ import print_function
import os
import re
import json
import errno
import numbers
import numpy as np

from io import open
from time import time
from tqdm import tqdm
from math import pow, tanh, log
from sklearn.feature_extraction.text import CountVectorizer
from util import is_a_collection, Print, VERBOSITY, Preproc as Pp

# python 2 and 3 compatibility
from functools import reduce
from six.moves import xrange

__version__ = "0.6.4"

ENCODING = "utf-8"

PARA_DELTR = "\n"   "字符串处理"
SENT_DELTR = r"\."
WORD_DELTR = r"\s"
WORD_REGEX = r"\w+(?:'\w+)?"

STR_UNKNOWN, STR_MOST_PROBABLE = "unknown", "most-probable"
STR_OTHERS_CATEGORY = "[others]"
STR_UNKNOWN_CATEGORY = "[unknown]"
IDX_UNKNOWN_CATEGORY = -1
STR_UNKNOWN_WORD = ''
IDX_UNKNOWN_WORD = -1
STR_VANILLA, STR_XAI = "vanilla", "xai"
STR_GV, STR_NORM_GV, STR_NORM_GV_XAI = "gv", "norm_gv", "norm_gv_xai"

STR_MODEL_FOLDER = "ss3_models"
STR_MODEL_EXT = "ss3m"

WEIGHT_SCHEMES_SS3 = ['only_cat', 'diff_all', 'diff_max', 'diff_median', 'diff_mean']
WEIGHT_SCHEMES_TF = ['binary', 'raw_count', 'term_freq', 'log_norm', 'double_norm']

VERBOSITY = VERBOSITY  # to allow "from pyss3 import VERBOSITY"

NAME = 0
VOCAB = 1

NEXT = 0
FR = 1
CV = 2
SG = 3
GV = 4
LV = 5
EMPTY_WORD_INFO = [0, 0, 0, 0, 0, 0]

NOISE_FR = 1
MIN_MAD_SD = .03


class SS3:
    """
    :param s: （“平滑度”（sigma）超参数值）
    :type s: float
    :param l: （显著性”（λ）超参数值）
    :type l: float
    :param p: “制裁”（rho）超参数值）
    :type p: float
    :param a: （alpha超参数值（即分类期间，小于alpha的置信值（cv）将被忽略））
    :type a: float
    :param name: （模型名称（用于从磁盘保存和加载模型））
    :type name: str
    :param cv_m: 用于计算每个参数的置信值（cv）的方法（单词或n-grams），选项包括："norm_gv_xai", "norm_gv" and "gv" (默认值: "norm_gv_xai")）
                 term (word or n-grams), options are:

    :type cv_m: str
    :param sg_m: 用于计算显著性（sg）函数的方法，选项是：“vanilla”和“xai”（默认值：“xai“））
    :type sg_m: str
    """


    __name__ = "model"
    __models_folder__ = STR_MODEL_FOLDER
    "超参数初始值"
    __s__ = .45
    __l__ = .5
    __p__ = 1
    __a__ = .0

    __multilabel__ = False

    __l_update__ = None
    __s_update__ = None
    __p_update__ = None

    __cv_cache__ = None
    __last_x_test__ = None
    __last_x_test_idx__ = None

    __prun_floor__ = 10
    __prun_trigger__ = 1000000
    __prun_counter__ = 0

    __zero_cv__ = None

    __parag_delimiter__ = PARA_DELTR
    __sent_delimiter__ = SENT_DELTR
    __word_delimiter__ = WORD_DELTR
    __word_regex__ = WORD_REGEX

    def __init__(
        self, s=None, l=None, p=None, a=None,
        name="", cv_m=STR_NORM_GV_XAI, sg_m=STR_XAI
    ):
        """
        类构造函数。
        """
        self.__name__ = (name or self.__name__).lower()

        self.__s__ = self.__s__ if s is None else s
        self.__l__ = self.__l__ if l is None else l
        self.__p__ = self.__p__ if p is None else p
        self.__a__ = self.__a__ if a is None else a

        try:
            float(self.__s__ + self.__l__ + self.__p__ + self.__a__)
        except BaseException:
            raise ValueError("hyperparameter values must be numbers")

        self.__categories_index__ = {}  #类别目录
        self.__categories__ = []        #类别
        self.__max_fr__ = []
        self.__max_gv__ = []            #最大置信度

        self.__index_to_word__ = {}     #目录到单词
        self.__word_to_index__ = {}     #单词到目录

        if cv_m == STR_NORM_GV_XAI:
            self.__cv__ = self.__cv_norm_gv_xai__  #置信度赋默认值
        elif cv_m == STR_NORM_GV:
            self.__cv__ = self.__cv_norm_gv__
        elif cv_m == STR_GV:
            self.__cv__ = self.__gv__

        if sg_m == STR_XAI:
            self.__sg__ = self.__sg_xai__
        elif sg_m == STR_VANILLA:
            self.__sg__ = self.__sg_vanilla__

        self.__cv_mode__ = cv_m
        self.__sg_mode__ = sg_m

        self.original_sumop_ngrams = self.summary_op_ngrams  #改变超参数
        self.original_sumop_sentences = self.summary_op_sentences
        self.original_sumop_paragraphs = self.summary_op_paragraphs

    def __lv__(self, ngram, icat, cache=True):
        """局部值函数"""
        if cache:
            return self.__trie_node__(ngram, icat)[LV]
        else:
            try:
                ilength = len(ngram) - 1
                fr = self.__trie_node__(ngram, icat)[FR]
                if fr > NOISE_FR:
                    max_fr = self.__max_fr__[icat][ilength]
                    local_value = (fr / float(max_fr)) ** self.__s__
                    return local_value
                else:
                    return 0
            except TypeError:
                return 0
            except IndexError:
                return 0

    def __sn__(self, ngram, icat):
        """制裁（sn）功能。"""
        m_values = [
            self.__sg__(ngram, ic)
            for ic in xrange(len(self.__categories__)) if ic != icat
        ]

        c = len(self.__categories__)

        s = sum([min(v, 1) for v in m_values])

        try:
            return pow((c - (s + 1)) / ((c - 1) * (s + 1)), self.__p__)
        except ZeroDivisionError:  # if c <= 1
            return 1.

    def __sg_vanilla__(self, ngram, icat, cache=True):
        """ 显著性（sg）函数定义。"""
        try:
            if cache:
                return self.__trie_node__(ngram, icat)[SG]
            else:
                ncats = len(self.__categories__)
                l = self.__l__
                lvs = [self.__lv__(ngram, ic) for ic in xrange(ncats)]
                lv = lvs[icat]

                M, sd = mad(lvs, ncats)

                if not sd and lv:
                    return 1.
                else:
                    return sigmoid(lv - M, l * sd)
        except TypeError:
            return 0.

    def __sg_xai__(self, ngram, icat, cache=True):
        """
        显著性（sn）函数的变体。此版本的sg函数为改进视觉解释。
        """
        try:
            if cache:
                return self.__trie_node__(ngram, icat)[SG]
            else:
                ncats = len(self.__categories__)
                l = self.__l__

                lvs = [self.__lv__(ngram, ic) for ic in xrange(ncats)]
                lv = lvs[icat]

                M, sd = mad(lvs, ncats)

                if l * sd <= MIN_MAD_SD:
                    sd = MIN_MAD_SD / l if l else 0

                # stopwords filter
                stopword = (M > .2) or (
                    sum(map(lambda v: v > 0.09, lvs)) == ncats
                )
                if (stopword and sd <= .1) or (M >= .3):
                    return 0.

                if not sd and lv:
                    return 1.

                return sigmoid(lv - M, l * sd)
        except TypeError:
            return 0.

    def __gv__(self, ngram, icat, cache=True):
        """
        （全局值（gv）函数。这是计算置信值（cv）的原始方法）
        """
        if cache:
            return self.__trie_node__(ngram, icat)[GV]
        else:
            lv = self.__lv__(ngram, icat)
            weight = self.__sg__(ngram, icat) * self.__sn__(ngram, icat)
            return lv * weight

    def __cv_norm_gv__(self, ngram, icat, cache=True):
        """
        计算术语置信值（cv）的替代方法。此变体规范化gv值，并将该值用作cv。
        """
        try:
            if cache:
                return self.__trie_node__(ngram, icat)[CV]
            else:
                try:
                    cv = self.__gv__(ngram, icat)
                    return cv / self.__max_gv__[icat][len(ngram) - 1]
                except (ZeroDivisionError, IndexError):
                    return .0

        except TypeError:
            return 0

    def __cv_norm_gv_xai__(self, ngram, icat, cache=True):
        """
        计算置信值（cv）的替代方法。这种变化不仅规范了gv值，还改进了视觉解释。
        """
        try:
            if cache:
                return self.__trie_node__(ngram, icat)[CV]
            else:
                try:
                    max_gv = self.__max_gv__[icat][len(ngram) - 1]
                    if (len(ngram) > 1):
                        # stopwords guard
                        n_cats = len(self.__categories__)
                        cats = xrange(n_cats)
                        sum_words_gv = sum([
                            self.__gv__([w], ic) for w in ngram for ic in cats
                        ])
                        if (sum_words_gv < .05):
                            return .0
                        elif len([
                            w for w in ngram
                            if self.__gv__([w], icat) >= .01
                        ]) == len(ngram):
                            gv = self.__gv__(ngram, icat)
                            return gv / max_gv + sum_words_gv
                            # return gv / max_gv * len(ngram)

                    gv = self.__gv__(ngram, icat)
                    return gv / max_gv
                except (ZeroDivisionError, IndexError):
                    return .0

        except TypeError:
            return 0

    def __apply_fn__(self, fn, ngram, cat=None):
        """gv、lv、sn、sg函数使用的私有方法。"""
        if ngram.strip() == '':
            return 0

        ngram = [self.get_word_index(w)
                 for w in re.split(self.__word_delimiter__, ngram)
                 if w]

        if cat is None:
            return fn(ngram) if IDX_UNKNOWN_WORD not in ngram else 0

        icat = self.get_category_index(cat)
        if icat == IDX_UNKNOWN_CATEGORY:
            raise InvalidCategoryError
        return fn(ngram, icat) if IDX_UNKNOWN_WORD not in ngram else 0

    def __summary_ops_are_pristine__(self):
        """如果摘要运算符未更改，则返回True。"""
        return self.original_sumop_ngrams == self.summary_op_ngrams and \
            self.original_sumop_sentences == self.summary_op_sentences and \
            self.original_sumop_paragraphs == self.summary_op_paragraphs

    def __classify_ngram__(self, ngram):
        """对给定的n-gram进行分类。"""
        cv = [
            self.__cv__(ngram, icat)
            for icat in xrange(len(self.__categories__))
        ]
        cv[:] = [(v if v > self.__a__ else 0) for v in cv]
        return cv

    def __classify_sentence__(self, sent, prep, json=False, prep_func=None):
        """把给定的句子分类。"""
        classify_trans = self.__classify_ngram__
        categories = self.__categories__
        cats = xrange(len(categories))
        word_index = self.get_word_index#索引
        word_delimiter = self.__word_delimiter__#分隔符
        word_regex = self.__word_regex__#正则表达式

        if not json:
            if prep or prep_func is not None:
                prep_func = prep_func or Pp.clean_and_ready
                sent = prep_func(sent)
            sent_words = [
                (w, w)
                for w in re_split_keep(word_regex, sent)
                if w
            ]
        else:
            if prep or prep_func is not None:
                sent_words = [
                    (w, Pp.clean_and_ready(w, dots=False) if prep_func is None else prep_func(w))
                    for w in re_split_keep(word_regex, sent)
                    if w
                ]
            else:
                sent_words = [
                    (w, w)
                    for w in re_split_keep(word_regex, sent)
                    if w
                ]

        if not sent_words:
            sent_words = [(u'.', u'.')]

        sent_iwords = [word_index(w) for _, w in sent_words]
        sent_len = len(sent_iwords)
        sent_parsed = []
        wcur = 0
        while wcur < sent_len:
            cats_ngrams_cv = [[0] for icat in cats]
            cats_ngrams_offset = [[0] for icat in cats]
            cats_ngrams_iword = [[-1] for icat in cats]
            cats_max_cv = [.0 for icat in cats]

            for icat in cats:
                woffset = 0
                word_raw = sent_words[wcur + woffset][0]
                wordi = sent_iwords[wcur + woffset]
                word_info = categories[icat][VOCAB]

                if wordi in word_info:
                    cats_ngrams_cv[icat][0] = word_info[wordi][CV]
                    word_info = word_info[wordi][NEXT]
                cats_ngrams_iword[icat][0] = wordi
                cats_ngrams_offset[icat][0] = woffset

                # 如果它是一个习得单词（对于这个类别来说不是未知和可见的），
                # 然后也试着识别学习过的n-gram
                if wordi != IDX_UNKNOWN_WORD and wordi in categories[icat][VOCAB]:
                    # while单词或单词分隔符（例如空格）
                    while wordi != IDX_UNKNOWN_WORD or re.match(word_delimiter, word_raw):
                        woffset += 1
                        if wcur + woffset >= sent_len:
                            break

                        word_raw = sent_words[wcur + woffset][0]
                        wordi = sent_iwords[wcur + woffset]

                        # if word is a word:
                        if wordi != IDX_UNKNOWN_WORD:
                            # if this word belongs to this category
                            if wordi in word_info:
                                cats_ngrams_cv[icat].append(word_info[wordi][CV])
                                cats_ngrams_iword[icat].append(wordi)
                                cats_ngrams_offset[icat].append(woffset)
                                word_info = word_info[wordi][NEXT]
                            else:
                                break

                    cats_max_cv[icat] = (max(cats_ngrams_cv[icat])
                                         if cats_ngrams_cv[icat] else .0)

            max_gv = max(cats_max_cv)
            use_ngram = True
            if (max_gv > self.__a__):
                icat_max_gv = cats_max_cv.index(max_gv)
                ngram_max_gv = cats_ngrams_cv[icat_max_gv].index(max_gv)
                offset_max_gv = cats_ngrams_offset[icat_max_gv][ngram_max_gv] + 1

                max_gv_sum_1_grams = max([
                    sum([
                        (categories[ic][VOCAB][wi][CV]
                         if wi in categories[ic][VOCAB]
                         else 0)
                        for wi
                        in cats_ngrams_iword[ic]
                    ])
                    for ic in cats
                ])

                if (max_gv_sum_1_grams > max_gv):
                    use_ngram = False
            else:
                use_ngram = False

            if not use_ngram:
                offset_max_gv = 1
                icat_max_gv = 0
                ngram_max_gv = 0

            sent_parsed.append(
                (
                    u"".join([raw_word for raw_word, _ in sent_words[wcur:wcur + offset_max_gv]]),
                    cats_ngrams_iword[icat_max_gv][:ngram_max_gv + 1]
                )
            )
            wcur += offset_max_gv

        get_word = self.get_word
        if not json:
            words_cvs = [classify_trans(seq) for _, seq in sent_parsed]
            if words_cvs:
                return self.summary_op_ngrams(words_cvs)
            return self.__zero_cv__
        else:
            get_tip = self.__trie_node__
            local_value = self.__lv__
            info = [
                {
                    "token": u"→".join(map(get_word, sequence)),
                    "lexeme": raw_sequence,
                    "cv": classify_trans(sequence),
                    "lv": [local_value(sequence, ic) for ic in cats],
                    "fr": [get_tip(sequence, ic)[FR] for ic in cats]
                }
                for raw_sequence, sequence in sent_parsed
            ]
            return {
                "words": info,
                "cv": self.summary_op_ngrams([v["cv"] for v in info]),
                "wmv": reduce(vmax, [v["cv"] for v in info])  # word max value
            }

    def __classify_paragraph__(self, parag, prep, json=False, prep_func=None):
        """对给定段落进行分类。"""
        if not json:
            sents_cvs = [
                self.__classify_sentence__(sent, prep=prep, prep_func=prep_func)
                for sent in re.split(self.__sent_delimiter__, parag)
                if sent
            ]
            if sents_cvs:
                return self.summary_op_sentences(sents_cvs)
            return self.__zero_cv__
        else:
            info = [
                self.__classify_sentence__(sent, prep=prep, prep_func=prep_func, json=True)
                for sent in re_split_keep(self.__sent_delimiter__, parag)
                if sent
            ]
            if info:
                sents_cvs = [v["cv"] for v in info]
                cv = self.summary_op_sentences(sents_cvs)
                wmv = reduce(vmax, [v["wmv"] for v in info])
            else:
                cv = self.__zero_cv__
                wmv = cv
            return {
                "sents": info,
                "cv": cv,
                "wmv": wmv  # word max value
            }

    def __trie_node__(self, ngram, icat):
        """获取此n-gram的单词查找树节点。"""
        try:
            word_info = self.__categories__[icat][VOCAB][ngram[0]]
            for word in ngram[1:]:
                word_info = word_info[NEXT][word]
            return word_info
        except BaseException:
            return EMPTY_WORD_INFO

    def __get_category__(self, name):
        """
        给定类别名称，返回类别数据。如果类别名称不存在，则创建一个新的类别名称。
        """
        try:
            return self.__categories_index__[name]
        except KeyError:
            self.__max_fr__.append([])
            self.__max_gv__.append([])
            self.__categories_index__[name] = len(self.__categories__)
            self.__categories__.append([name, {}])  # name, vocabulary
            self.__zero_cv__ = (0,) * len(self.__categories__)
            return self.__categories_index__[name]

    def __get_category_length__(self, icat):
        """
        返回类别长度。类别长度是训练期间看到的单词总数。
        """
        size = 0
        vocab = self.__categories__[icat][VOCAB]
        for word in vocab:
            size += vocab[word][FR]
        return size

    def __get_most_probable_category__(self):
        """返回最可能类别的索引"""
        sizes = []
        for icat in xrange(len(self.__categories__)):
            sizes.append((icat, self.__get_category_length__(icat)))
        return sorted(sizes, key=lambda v: v[1])[-1][0]

    def __get_vocabularies__(self, icat, vocab, preffix, n_grams, output, ngram_char="_"):
        """获取包含信息的n-grams类别列表。"""
        senq_ilen = len(preffix)
        get_name = self.get_word

        seq = preffix + [None]
        if len(seq) > n_grams:
            return

        for word in vocab:
            seq[-1] = word
            if (self.__cv__(seq, icat) > 0):
                output[senq_ilen].append(
                    (
                        ngram_char.join([get_name(wi) for wi in seq]),
                        vocab[word][FR],
                        self.__gv__(seq, icat),
                        self.__cv__(seq, icat)
                    )
                )
            self.__get_vocabularies__(
                icat, vocab[word][NEXT], seq, n_grams, output, ngram_char
            )

    def __get_category_vocab__(self, icat):
        """获取按置信值排序的n-gram的类别列表。"""
        category = self.__categories__[icat]
        vocab = category[VOCAB]
        w_seqs = ([w] for w in vocab)

        vocab_icat = (
            (
                self.get_word(wseq[0]),
                vocab[wseq[0]][FR],
                self.__lv__(wseq, icat),
                self.__gv__(wseq, icat),
                self.__cv__(wseq, icat)
            )
            for wseq in w_seqs if self.__gv__(wseq, icat) > self.__a__
        )
        return sorted(vocab_icat, key=lambda k: -k[-1])

    def __get_def_cat__(self, def_cat):
        """给定`def_cat`参数，获取默认类别值。"""
        if def_cat is not None and (def_cat not in [STR_MOST_PROBABLE, STR_UNKNOWN] and
                                    self.get_category_index(def_cat) == IDX_UNKNOWN_CATEGORY):
            raise ValueError(
                "the default category must be 'most-probable', 'unknown', or a category name"
                " (current value is '%s')." % str(def_cat)
            )
        def_cat = None if def_cat == STR_UNKNOWN else def_cat
        return self.get_most_probable_category() if def_cat == STR_MOST_PROBABLE else def_cat

    def __get_next_iwords__(self, sent, icat):
        """返回可能的后续单词索引列表。"""
        if not self.get_category_name(icat):
            return []

        vocab = self.__categories__[icat][VOCAB]
        word_index = self.get_word_index
        sent = Pp.clean_and_ready(sent)
        sent = [
            word_index(w)
            for w in sent.strip(".").split(".")[-1].split(" ") if w
        ]

        tips = []
        for word in sent:
            if word is None:
                tips[:] = []
                continue

            tips.append(vocab)

            tips[:] = (
                tip[word][NEXT]
                for tip in tips if word in tip and tip[word][NEXT]
            )

        if len(tips) == 0:
            return []

        next_words = tips[0]
        next_nbr_words = float(sum([next_words[w][FR] for w in next_words]))
        return sorted(
            [
                (
                    word1,
                    next_words[word1][FR],
                    next_words[word1][FR] / next_nbr_words
                )
                for word1 in next_words
            ],
            key=lambda k: -k[1]
        )

    def __prune_cat_trie__(self, vocab, prune=False, min_n=None):
        """删减给定类别的trie单词查找树。"""
        prun_floor = min_n or self.__prun_floor__
        remove = []
        for word in vocab:
            if prune and vocab[word][FR] <= prun_floor:
                vocab[word][NEXT] = None
                remove.append(word)
            else:
                self.__prune_cat_trie__(vocab[word][NEXT], prune=True)

        for word in remove:
            del vocab[word]

    def __prune_tries__(self):
        """删减每个类别的单词查找树"""
        Print.info("pruning tries...", offset=1)
        for category in self.__categories__:
            self.__prune_cat_trie__(category[VOCAB])
        self.__prun_counter__ = 0

    def __cache_lvs__(self, icat, vocab, preffix):
        """缓存所有局部值"""
        for word in vocab:
            sequence = preffix + [word]
            vocab[word][LV] = self.__lv__(sequence, icat, cache=False)
            self.__cache_lvs__(icat, vocab[word][NEXT], sequence)

    def __cache_gvs__(self, icat, vocab, preffix):
        """缓存所有全局值"""
        for word in vocab:
            sequence = preffix + [word]
            vocab[word][GV] = self.__gv__(sequence, icat, cache=False)
            self.__cache_gvs__(icat, vocab[word][NEXT], sequence)

    def __cache_sg__(self, icat, vocab, preffix):
        """缓存所有显著性权重值"""
        for word in vocab:
            sequence = preffix + [word]
            vocab[word][SG] = self.__sg__(sequence, icat, cache=False)
            self.__cache_sg__(icat, vocab[word][NEXT], sequence)

    def __cache_cvs__(self, icat, vocab, preffix):
        """缓存所有置信值"""
        for word in vocab:
            sequence = preffix + [word]
            vocab[word][CV] = self.__cv__(sequence, icat, False)
            self.__cache_cvs__(icat, vocab[word][NEXT], sequence)

    def __update_max_gvs__(self, icat, vocab, preffix):
        """更新所有最大全局值"""
        gv = self.__gv__
        max_gvs = self.__max_gv__[icat]
        sentence_ilength = len(preffix)

        sequence = preffix + [None]
        for word in vocab:
            sequence[-1] = word
            sequence_gv = gv(sequence, icat)
            if sequence_gv > max_gvs[sentence_ilength]:
                max_gvs[sentence_ilength] = sequence_gv
            self.__update_max_gvs__(icat, vocab[word][NEXT], sequence)

    def __update_needed__(self):
        """（参数更新函数）如果需要更新，则返回True，否则返回false。"""
        return (self.__s__ != self.__s_update__ or
                self.__l__ != self.__l_update__ or
                self.__p__ != self.__p_update__)

    def __save_cat_vocab__(self, icat, path, n_grams):
        """将类别词汇保存在``path``中。"""
        if n_grams == -1:
            n_grams = 20  # infinite

        category = self.__categories__[icat]
        cat_name = self.get_category_name(icat)
        vocab = category[VOCAB]
        vocabularies_out = [[] for _ in xrange(n_grams)]

        terms = ["words", "bigrams", "trigrams"]

        self.__get_vocabularies__(icat, vocab, [], n_grams, vocabularies_out)

        Print.info("saving '%s' vocab" % cat_name)

        for ilen in xrange(n_grams):
            if vocabularies_out[ilen]:
                term = terms[ilen] if ilen <= 2 else "%d-grams" % (ilen + 1)
                voc_path = os.path.join(
                    path, "ss3_vocab_%s(%s).csv" % (cat_name, term)
                )
                f = open(voc_path, "w+", encoding=ENCODING)
                vocabularies_out[ilen].sort(key=lambda k: -k[-1])
                f.write(u"%s,%s,%s,%s\n" % ("term", "fr", "gv", "cv"))
                for trans in vocabularies_out[ilen]:
                    f.write(u"%s,%d,%f,%f\n" % tuple(trans))
                f.close()
                Print.info("\t[ %s stored in '%s'" % (term, voc_path))

    def __update_cv_cache__(self):
        """更新置信值缓存"""
        if self.__cv_cache__ is None:
            self.__cv_cache__ = np.zeros((len(self.__index_to_word__), len(self.__categories__)))
        cv = self.__cv__
        for term_idx, cv_vec in enumerate(self.__cv_cache__):
            for cat_idx, _ in enumerate(cv_vec):
                try:
                    cv_vec[cat_idx] = cv([term_idx], cat_idx)
                except KeyError:
                    cv_vec[cat_idx] = 0

    def __predict_fast__(
        self, x_test, def_cat=STR_MOST_PROBABLE, labels=True,
        multilabel=False, proba=False, prep=True, leave_pbar=True
    ):
        """“predict”方法的更快版本（使用numpy）。"""
        if not def_cat or def_cat == STR_UNKNOWN:
            def_cat = IDX_UNKNOWN_CATEGORY
        elif def_cat == STR_MOST_PROBABLE:
            def_cat = self.__get_most_probable_category__()
        else:
            def_cat = self.get_category_index(def_cat)
            if def_cat == IDX_UNKNOWN_CATEGORY:
                raise InvalidCategoryError

        # does the special "[others]" category exist? (only used in multilabel classification)
        __other_idx__ = self.get_category_index(STR_OTHERS_CATEGORY)

        if self.__update_needed__():
            self.update_values()

        if self.__cv_cache__ is None:
            self.__update_cv_cache__()
            self.__last_x_test__ = None  # could have learned a new word (in `learn`)
        cv_cache = self.__cv_cache__

        x_test_hash = list_hash(x_test)
        if x_test_hash == self.__last_x_test__:
            x_test_idx = self.__last_x_test_idx__
        else:
            self.__last_x_test__ = x_test_hash
            self.__last_x_test_idx__ = [None] * len(x_test)
            x_test_idx = self.__last_x_test_idx__
            word_index = self.get_word_index
            for doc_idx, doc in enumerate(tqdm(x_test, desc="Caching documents",
                                               leave=False, disable=Print.is_quiet())):
                x_test_idx[doc_idx] = [
                    word_index(w)
                    for w
                    in re.split(self.__word_delimiter__, Pp.clean_and_ready(doc) if prep else doc)
                    if word_index(w) != IDX_UNKNOWN_WORD
                ]

        y_pred = [None] * len(x_test)
        for doc_idx, doc in enumerate(tqdm(x_test_idx, desc="Classification",
                                           leave=leave_pbar, disable=Print.is_quiet())):
            if self.__a__ > 0:
                doc_cvs = cv_cache[doc]
                doc_cvs[doc_cvs <= self.__a__] = 0
                pred_cv = np.add.reduce(doc_cvs, 0)
            else:
                pred_cv = np.add.reduce(cv_cache[doc], 0)

            if proba:
                y_pred[doc_idx] = list(pred_cv)
                continue

            if not multilabel:
                if pred_cv.sum() == 0:
                    y_pred[doc_idx] = def_cat
                else:
                    y_pred[doc_idx] = np.argmax(pred_cv)

                if labels:
                    if y_pred[doc_idx] != IDX_UNKNOWN_CATEGORY:
                        y_pred[doc_idx] = self.__categories__[y_pred[doc_idx]][NAME]
                    else:
                        y_pred[doc_idx] = STR_UNKNOWN_CATEGORY
            else:
                if pred_cv.sum() == 0:
                    if def_cat == IDX_UNKNOWN_CATEGORY:
                        y_pred[doc_idx] = []
                    else:
                        y_pred[doc_idx] = [self.get_category_name(def_cat) if labels else def_cat]
                else:
                    r = sorted([(i, pred_cv[i])
                                for i in range(pred_cv.size)],
                               key=lambda e: -e[1])
                    if labels:
                        y_pred[doc_idx] = [self.get_category_name(cat_i)
                                           for cat_i, _ in r[:kmean_multilabel_size(r)]]
                    else:
                        y_pred[doc_idx] = [cat_i for cat_i, _ in r[:kmean_multilabel_size(r)]]

                # if the special "[others]" category exists
                if __other_idx__ != IDX_UNKNOWN_CATEGORY:
                    # if its among the predicted labels, remove (hide) it
                    if labels:
                        if STR_OTHERS_CATEGORY in y_pred[doc_idx]:
                            y_pred[doc_idx].remove(STR_OTHERS_CATEGORY)
                    else:
                        if __other_idx__ in y_pred[doc_idx]:
                            y_pred[doc_idx].remove(__other_idx__)

        return y_pred

    def summary_op_ngrams(self, cvs):
        """
         n元置信向量的汇总运算符。

        默认情况下，它返回所有置信度的添加矢量。但是，如果您想使用自定义摘要运算符，必须替换此函数
        如下例所示：

            >>> def my_summary_op(cvs):
            >>>     return cvs[0]
            >>> ...
            >>> clf = SS3()
            >>> ...
            >>> clf.summary_op_ngrams = my_summary_op

        任何接收矢量列表和可以使用返回单个矢量。在上面的例子中摘要运算符被用户定义的
        ``my_summary_op``忽略所有置信向量仅返回第一个n-gram的置信向量（这除了是一个说明性的例子外，没有任何实际意义）。）

        :param cvs: （n-gram置信向量列表）
        :type cvs: （浮点数列表）
        :returns: 句子置信向量）
        :rtype: list (of float)
        """
        return reduce(vsum, cvs)

    def summary_op_sentences(self, cvs):
        """
        句子置信向量的摘要运算符）

        :param cvs:（列出句子置信向量）
        :type cvs: （浮点数列表）
        :returns: 段落置信向量）
        :rtype: list (of float)
        """
        return reduce(vsum, cvs)

    def summary_op_paragraphs(self, cvs):
        """
       段落置信向量的汇总运算符。


        :param cvs: 段落置信向量列表
        :type cvs: list (of list of float)
        :returns: 文档置信向量
        :rtype: list (of float)
        """
        return reduce(vsum, cvs)

    def get_name(self):
        """
        返回模型名称

        :returns: the model's name.
        :rtype: str
        """
        return self.__name__

    def set_name(self, name):
        """
        设置模型名称）

        :param name: the model's name.
        :type name: str
        """
        self.__name__ = name

    def set_hyperparameters(self, s=None, l=None, p=None, a=None):
        """
        设置超参数
        """
        if s is not None:
            self.set_s(s)
        if l is not None:
            self.set_l(l)
        if p is not None:
            self.set_p(p)
        if a is not None:
            self.set_a(a)

    def get_hyperparameters(self):
        """
       获得超参数值

        :returns: 返回具有超参数当前值的元组
        :rtype: tuple
        """
        return self.__s__, self.__l__, self.__p__, self.__a__

    def set_model_path(self, path):
        """
        Overwrite the default path from which the model will be loaded (or saved to).

        Note: be aware that the PySS3 Command Line tool looks for
        a local folder called ``ss3_models`` to load models.
        Therefore, the ``ss3_models`` folder will be always automatically
        append to the given ``path`` (e.g. if ``path="my/path/"``, it will
        be converted into ``my/path/ss3_models``).

        :param path: the path
        :type path: str
        """
        self.__models_folder__ = os.path.join(path, STR_MODEL_FOLDER)

    def set_block_delimiters(self, parag=None, sent=None, word=None):
        r"""覆盖用于将输入文档拆分为块的默认分隔符。）
            分隔符是从简单的（例如``“”``）到更复杂的（例如`r“[^\s\w\d]”``）。注意：记住正则表达式有一些保留字符，
        例如，点（.），在这种情况下，使用反斜杠表示引用字符本身（例如``\.``）
        e.g.

        >>> ss3.set_block_delimiters(word="\s"),,
        >>> ss3.set_block_delimiters(word="\s", parag="\n\n")
        >>> ss3.set_block_delimiters(parag="\n---\n")
        >>> ss3.set_block_delimiters(sent="\.")
        >>> ss3.set_block_delimiters(word="\|")
        >>> ss3.set_block_delimiters(word=" ")

        :param parag: （段落新分隔符）
        :type parag: str
        :param sent: 句子新分隔符）
        :type sent: str
        :param word: 单词新分隔符）
        :type word: str
        """
        if parag:
            self.set_delimiter_paragraph(parag)
        if sent:
            self.set_delimiter_sentence(sent)
        if word:
            self.set_delimiter_word(word)

    def set_delimiter_paragraph(self, regex):
        r"""
        设置用于将文档拆分为段落的分隔符。
        请记住，正则表达式有某些保留字符，例如，点（.），
        在这种情况下，使用反斜杠表示引用字符本身（例如``\.``）
        :param regex: 新分隔符的正则表达式）
        :type regex: str
        """
        self.__parag_delimiter__ = regex

    def set_delimiter_sentence(self, regex):
        r"""
       设置用于将文档拆分为句子的分隔符。）
        （请记住，正则表达式有某些保留字符，例如，点（.），在这种情况下，使用反斜杠表示引用字符本身（例如``\.``））
        :param regex: 新分隔符的正则表达式）
        :type regex: str
        """
        self.__sent_delimiter__ = regex

    def set_delimiter_word(self, regex):
        r"""
       （设置用于将文档拆分为单词的分隔符。）
        :param regex: 新分隔符的正则表达式）
        :type regex: str
        """
        self.__word_delimiter__ = regex

    def set_s(self, value):
        """
        设置“平滑度”（sigma）超参数值。

        :param value: （超参数值）
        :type value: float
        """
        self.__s__ = float(value)

    def get_s(self):
        """
       获取“平滑度”（sigma）超参数值。

        :returns: the hyperparameter value
        :rtype: float
        """
        return self.__s__

    def set_l(self, value):
        """
        设置“重要性”参数

        :param value: the hyperparameter value
        :type value: float
        """
        self.__l__ = float(value)

    def get_l(self):
        """
        获取“重要性”参数的值

        :returns: the hyperparameter value
        :rtype: float
        """
        return self.__l__

    def set_p(self, value):
        """
        设置“惩罚”参数的值

        :param value: the hyperparameter value
        :type value: float
        """
        self.__p__ = float(value)

    def get_p(self):
        """
       获取“惩罚”参数的值

        :returns: the hyperparameter value
        :rtype: float
        """
        return self.__p__

    def set_a(self, value):
        """
       设置alpha超参数值。

        (置信值（cv）小于α的所有项,将在分类期间忽略。)
        :param value: the hyperparameter value
        :type value: float
        """
        self.__a__ = float(value)

    def get_a(self):
        """
        获得alpha超参数的值

        :returns: the hyperparameter value
        :rtype: float
        """
        return self.__a__

    def get_categories(self, all=False):
        """
        获取类别名称列表

        :returns:类别名称列表
        :rtype: list (of str)
        """
        return [
            self.get_category_name(ci)
            for ci in range(len(self.__categories__))
            if all or self.get_category_name(ci) != STR_OTHERS_CATEGORY
        ]

    def get_most_probable_category(self):
        """
        获取可能类别的名称

        :returns:最可能类别的名称
        :rtype: str
        """
        return self.get_category_name(self.__get_most_probable_category__())

    def get_ngrams_length(self):
        """
        返回学习的最长n-gram长度

        :returns: the length of longest learned n-gram.
        :rtype: int
        """
        return len(self.__max_fr__[0]) if len(self.__max_fr__) > 0 else 0

    def get_category_index(self, name):
        """
        给定其名称，返回类别索引

        :param name: （类型名称）
        :type name: str
        :returns: （类别索引）
        :rtype: int
        """
        try:
            return self.__categories_index__[name]
        except KeyError:
            return IDX_UNKNOWN_CATEGORY

    def get_category_name(self, index):
        """
        给定其索引，返回类别名称

        :param index: 类别索引
        :type index: int
        :returns:类别名称
        :rtype: str
        """
        try:
            if isinstance(index, list):
                index = index[0]
            return self.__categories__[index][NAME]
        except IndexError:
            return STR_UNKNOWN_CATEGORY

    def get_word_index(self, word):
        """
        给定一个词，返回索引

        :param name: a word
        :type name: str
        :returns: 单词索引
        :rtype: int
        """
        try:
            return self.__word_to_index__[word]
        except KeyError:
            return IDX_UNKNOWN_WORD

    def get_word(self, index):
        """
        给定索引，返回单词

        :param index: 单词索引
        :type index: int
        :returns: the word (or ``STR_UNKNOWN_WORD`` if the word doesn't exist).
        :rtype: int
        :rtype: str
        """
        return (
            self.__index_to_word__[index]
            if index in self.__index_to_word__ else STR_UNKNOWN_WORD
        )

    def get_next_words(self, sent, cat, n=None):
        """
       给定一个句子，返回``n``（可能）后面的单词列表。

        :param sent: 一个句子
        :type sent: str
        :param cat: 类别名称
        :type cat: str
        :param n:最大可能答案数
        :type n: int
        :returns: 元组列表（单词，频率，概率）
        :rtype: list (of tuple)
        :raises: InvalidCategoryError
        """
        icat = self.get_category_index(cat)

        if icat == IDX_UNKNOWN_CATEGORY:
            raise InvalidCategoryError

        guessedwords = [
            (self.get_word(iword), fr, P)
            for iword, fr, P in self.__get_next_iwords__(sent, icat) if fr
        ]
        if n is not None and guessedwords:
            return guessedwords[:n]
        return guessedwords

    def get_stopwords(self, sg_threshold=.01):
        """
       获取（可识别的）非索引词列表。

        :param sg_threshold:用作阈值的显著性（sg）将单词视为stopwords（即带有所有类别的sg<``sg_threshold``将被视为“关键词”）
        :type sg_threshold: float
        :returns:停止字列表
        :rtype: list (of str)
        """
        if not self.__categories__:
            return

        iwords = self.__index_to_word__
        sg_threshold = float(sg_threshold or .01)
        categories = self.__categories__
        cats_len = len(categories)
        sg = self.__sg__
        stopwords = []
        vocab = categories[0][VOCAB]

        for word0 in iwords:
            word_sg = [
                sg([word0], c_i)
                for c_i in xrange(cats_len)
            ]
            word_cats_len = len([v for v in word_sg if v < sg_threshold])
            if word_cats_len == cats_len:
                stopwords.append(word0)

        stopwords = [
            iwords[w0]
            for w0, v
            in sorted(
                [
                    (w0, vocab[w0][FR] if w0 in vocab else 0)
                    for w0 in stopwords
                ],
                key=lambda k: -k[1]
            )
        ]

        return stopwords

    def save_model(self, path=None):
        """
        将模型保存到磁盘

        :param path: the path to save the model to
        :type path: str

        :raises: OSError
        """
        if path:
            self.set_model_path(path)

        stime = time()
        Print.info(
            "saving model (%s/%s.%s)..."
            %
            (self.__models_folder__, self.__name__, STR_MODEL_EXT),
            False
        )
        json_file_format = {
            "__a__": self.__a__,
            "__l__": self.__l__,
            "__p__": self.__p__,
            "__s__": self.__s__,
            "__max_fr__": self.__max_fr__,
            "__max_gv__": self.__max_gv__,
            "__categories__": self.__categories__,
            "__categories_index__": self.__categories_index__,
            "__index_to_word__": self.__index_to_word__,
            "__word_to_index__": self.__word_to_index__,
            "__cv_mode__": self.__cv_mode__,
            "__sg_mode__": self.__sg_mode__,
            "__multilabel__": self.__multilabel__
        }

        try:
            os.makedirs(self.__models_folder__)
        except OSError as ose:
            if ose.errno == errno.EEXIST and os.path.isdir(self.__models_folder__):
                pass
            else:
                raise

        json_file = open(
            "%s/%s.%s" % (
                self.__models_folder__,
                self.__name__,
                STR_MODEL_EXT
            ), "w", encoding=ENCODING
        )

        try:  # python 3
            json_file.write(json.dumps(json_file_format))
        except TypeError:  # python 2
            json_file.write(json.dumps(json_file_format).decode(ENCODING))

        json_file.close()
        Print.info("(%.1fs)" % (time() - stime))

    def load_model(self, path=None):
        """
       从磁盘加载模型

        :param path: the path to load the model from
        :type path: str

        :raises: OSError
        """
        if path:
            self.set_model_path(path)

        stime = time()
        Print.info("loading '%s' model from disk..." % self.__name__)

        json_file = open(
            "%s/%s.%s" % (
                self.__models_folder__,
                self.__name__,
                STR_MODEL_EXT
            ), "r", encoding=ENCODING
        )
        jmodel = json.loads(json_file.read(), object_hook=key_as_int)
        json_file.close()

        self.__max_fr__ = jmodel["__max_fr__"]
        self.__max_gv__ = jmodel["__max_gv__"]
        self.__l__ = jmodel["__l__"]
        self.__p__ = jmodel["__p__"]
        self.__s__ = jmodel["__s__"]
        self.__a__ = jmodel["__a__"]
        self.__categories__ = jmodel["__categories__"]
        self.__categories_index__ = jmodel["__categories_index__"]
        self.__index_to_word__ = jmodel["__index_to_word__"]
        self.__word_to_index__ = jmodel["__word_to_index__"]
        self.__cv_mode__ = jmodel["__cv_mode__"]
        self.__multilabel__ = jmodel["__multilabel__"] if "__multilabel__" in jmodel else False
        self.__sg_mode__ = (jmodel["__sg_mode__"]
                            if "__sg_mode__" in jmodel
                            else jmodel["__sn_mode__"])

        self.__zero_cv__ = (0,) * len(self.__categories__)
        self.__s_update__ = self.__s__
        self.__l_update__ = self.__l__
        self.__p_update__ = self.__p__

        Print.info("(%.1fs)" % (time() - stime))

    def save_cat_vocab(self, cat, path="./", n_grams=-1):
        """
        将类别词汇保存到磁盘

        :param cat: 类别名称
        :type cat: str
        :param path: 存储词汇表路径
        :type path: str
        :param n_grams:表示要存储的n-grams（例如，仅1-grams，2、3等）。默认值-1存储所有学习的n-grams（1-grams、2-grams、3-grams等）
        :type n_grams: int
        :raises: InvalidCategoryError
        """
        if self.get_category_index(cat) == IDX_UNKNOWN_CATEGORY:
            raise InvalidCategoryError

        self.__save_cat_vocab__(self.get_category_index(cat), path, n_grams)

    def save_vocab(self, path="./", n_grams=-1):
        """
        将所学词汇保存到磁盘。

        :param path:存储词汇表的路径
        :type path: str
        :param n_grams: 表示要存储的n-grams（例如，仅1-grams，2、3等）。默认值-1存储所有学习的n-grams（1-grams、2-grams、3-grams等）
        :type n_grams: int
        """
        for icat in xrange(len(self.__categories__)):
            self.__save_cat_vocab__(icat, path, n_grams)

    def save_wordcloud(self, cat, top_n=100, n_grams=1, path=None, size=1024,
                       shape="circle", palette="cartocolors.qualitative.Prism_2", color=None,
                       background_color="white", plot=False):
        """
    创建一个单词云并将其作为图像保存到磁盘。单词云显示由模型学习的置信值选择的前n个单词。此外，单个单词的大小取决于所学的值。
        :param cat: the category label
        :type cat: str
        :param top_n: number of words to be taken into account.
                    For instance, top 50 words (default: 100).
        :type top_n: int
        :param n_grams: indicates the word n-grams to be used to create the cloud. For instance,
                        1 for word cloud, 2 for bigrams cloud, 3 for trigrams cloud, and so on
                        (default: 1).
        :type n_grams: int
        :param path: the path to the image file in which to store the word cloud
                     (e.g. "../../my_wordcloud.jpg").
                     If no path is given, by default, the image file will be stored in the current
                     working directory as "wordcloud_topN_CAT(NGRAM).png" where N is the `top_n`
                     value, CAT the category label and NGRAM indicates what n-grams  populate
                     the could.
        :type path: str
        :param size: the size of the image in pixels (default: 1024)
        :type size: int
        :param shape: the shape of the cloud (a FontAwesome icon name).
                      The complete list of allowed icon names are available at
                      https://fontawesome.com/v5.15/icons?d=gallery&p=1&m=free
                      (default: "circle")
        :type shape: str
        :param palette: the color palette used for coloring words by giving the
                        palettable module and palette name
                        (list available at https://jiffyclub.github.io/palettable/)
                        (default: "cartocolors.qualitative.Prism_2")
        :type palette: str
        :param color: a custom color for words (if given, overrides the color palette).
                      The color string could be the hex color code (e.g. "#FF5733") or the HTML
                      color name (e.g. "tomato"). The complete list of HTML color names is available
                      at https://www.w3schools.com/colors/colors_names.asp
        :type color: str
        :param background_color: the background color as either the HTML color name or the hex code
                                 (default: "white").
        :type background_color: str
        :param plot: whether or not to also plot the cloud (after saving the file)
                     (default: False)
        :type plot: bool
        :raises: InvalidCategoryError, ValueError
        """
        if self.get_category_index(cat) == IDX_UNKNOWN_CATEGORY:
            raise InvalidCategoryError

        if top_n < 1 or n_grams < 1 or size < 1:
            raise ValueError("`top_n`, `n_grams`, and `size` arguments must be positive integers")

        import stylecloud

        icat = self.get_category_index(cat)
        category = self.__categories__[icat]
        vocab = category[VOCAB]
        vocabularies_out = [[] for _ in xrange(n_grams)]

        self.__get_vocabularies__(icat, vocab, [], n_grams, vocabularies_out, "+")

        ilen = n_grams - 1

        if not vocabularies_out[ilen]:
            Print.info("\t[ empty word could: no %d-grams to be shown ]" % n_grams)
            return

        terms = dict((t, cv)
                     for t, _, _, cv
                     in sorted(vocabularies_out[ilen], key=lambda k: -k[-1])[:top_n])

        if path is None:
            term = ["", "(bigrams)", "(trigrams)"][ilen] if ilen <= 2 else "(%d-grams)" % (ilen + 1)
            path = "wordcloud_top%d_%s%s.png" % (top_n, cat, term)

        stylecloud.gen_stylecloud(
            terms,
            icon_name="fas fa-%s" % shape,
            output_name=path,
            palette=palette,
            colors=color,
            background_color=background_color,
            size=size
        )
        Print.info("\t[ word cloud stored in '%s' ]" % path)

        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            wc = mpimg.imread(path)

            plt.figure(figsize=(size / 100., size / 100.))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.show()

    def update_values(self, force=False):
        """
        更新模型值（cv、gv、lv等）。

        :param force: 强制更新（即使超参数没有更改）
        :type force: bool
        """
        update = 0
        if force or self.__s_update__ != self.__s__:
            update = 3
        elif self.__l_update__ != self.__l__:
            update = 2
        elif self.__p_update__ != self.__p__:
            update = 1

        if update == 0:
            Print.info("nothing to update...", offset=1)
            return

        category_len = len(self.__categories__)
        categories = xrange(category_len)
        category_names = [self.get_category_name(ic) for ic in categories]
        stime = time()
        Print.info("about to start updating values...", offset=1)
        if update == 3:  # only if s has changed
            Print.info("caching lv values", offset=1)
            for icat in categories:
                Print.info(
                    "lv values for %d (%s)" % (icat, category_names[icat]),
                    offset=4
                )
                self.__cache_lvs__(icat, self.__categories__[icat][VOCAB], [])

        if update >= 2:  # only if s or l have changed
            Print.info("caching sg values", offset=1)
            for icat in categories:
                Print.info(
                    "sg values for %d (%s)" % (icat, category_names[icat]),
                    offset=4
                )
                self.__cache_sg__(icat, self.__categories__[icat][VOCAB], [])

        Print.info("caching gv values")
        for icat in categories:
            Print.info(
                "gv values for %d (%s)" % (icat, category_names[icat]),
                offset=4
            )
            self.__cache_gvs__(icat, self.__categories__[icat][VOCAB], [])

        if self.__cv_mode__ != STR_GV:
            Print.info("updating max gv values", offset=1)
            for icat in categories:
                Print.info(
                    "max gv values for %d (%s)" % (icat, category_names[icat]),
                    offset=4
                )
                self.__max_gv__[icat] = list(
                    map(lambda _: 0, self.__max_gv__[icat])
                )
                self.__update_max_gvs__(
                    icat, self.__categories__[icat][VOCAB], []
                )

            Print.info("max gv values have been updated", offset=1)

            Print.info("caching confidence values (cvs)", offset=1)
            for icat in categories:
                Print.info(
                    "cvs for %d (%s)" % (icat, category_names[icat]),
                    offset=4
                )
                self.__cache_cvs__(icat, self.__categories__[icat][VOCAB], [])
        Print.info("finished --time: %.1fs" % (time() - stime), offset=1)

        self.__s_update__ = self.__s__
        self.__l_update__ = self.__l__
        self.__p_update__ = self.__p__

        if self.__cv_cache__ is not None:
            self.__update_cv_cache__()

    def print_model_info(self):
        """打印有关模型的信息"""
        print()
        print(" %s: %s\n" % (
            Print.style.green(Print.style.ubold("NAME")),
            Print.style.warning(self.get_name())
        ))

    def print_hyperparameters_info(self):
        """打印有关模型的超参数"""
        print()
        print(
            " %s:\n" % Print.style.green(Print.style.ubold("HYPERPARAMETERS"))
        )
        print("\tSmoothness(s):", Print.style.warning(self.__s__))
        print("\tSignificance(l):", Print.style.warning(self.__l__))
        print("\tSanction(p):", Print.style.warning(self.__p__))
        print("")
        print("\tAlpha(a):", Print.style.warning(self.__a__))

    def print_categories_info(self):
        """打印有关学习类别的信息"""
        if not self.__categories__:
            print(
                "\n %s: None\n"
                % Print.style.green(Print.style.ubold("CATEGORIES"))
            )
            return

        cat_len = max([
            len(self.get_category_name(ic))
            for ic in xrange(len(self.__categories__))
        ])
        cat_len = max(cat_len, 8)
        row_template = Print.style.warning("\t{:^%d} " % cat_len)
        row_template += "| {:^5} | {:^10} | {:^11} | {:^13} | {:^6} |"
        print()
        print("\n %s:\n" % Print.style.green(Print.style.ubold("CATEGORIES")))
        print(
            row_template
            .format(
                "Category", "Index", "Length",
                "Vocab. Size", "Word Max. Fr.", "N-gram"
            )
        )
        print(
            (
                "\t{:-<%d}-|-{:-<5}-|-{:-<10}-|-{:-<11}-|-{:-<13}-|-{:-<6}-|"
                % cat_len
            )
            .format('', '', '', '', '', '')
        )

        mpci = self.__get_most_probable_category__()
        mpc_size = 0
        mpc_total = 0
        for icat, category in enumerate(self.__categories__):
            icat_size = self.__get_category_length__(icat)
            print(
                row_template
                .format(
                    category[NAME],
                    icat, icat_size,
                    len(category[VOCAB]),
                    self.__max_fr__[icat][0],
                    len(self.__max_fr__[icat])
                )
            )

            mpc_total += icat_size
            if icat == mpci:
                mpc_size = icat_size

        print(
            "\n\t%s: %s %s"
            %
            (
                Print.style.ubold("Most Probable Category"),
                Print.style.warning(self.get_category_name(mpci)),
                Print.style.blue("(%.2f%%)" % (100.0 * mpc_size / mpc_total))
            )
        )
        print()

    def print_ngram_info(self, ngram):
        """
        打印有关给定n-gram的调试信息。

        即打印n-gram频率（fr）、局部值（lv）、全局值（gv）、置信值（cv）、制裁（sn）权重，显著性（sg）权重。
        :param ngram: the n-gram (e.g. "machine", "machine learning", etc.)
        :type ngram: str
        """
        if not self.__categories__:
            return

        word_index = self.get_word_index
        n_gram_str = ngram
        ngram = [word_index(w)
                 for w in re.split(self.__word_delimiter__, ngram)
                 if w]

        print()
        print(
            " %s: %s (%s)" % (
                Print.style.green(
                    "%d-GRAM" % len(ngram) if len(ngram) > 1 else "WORD"
                ),
                Print.style.warning(n_gram_str),
                "is unknown"
                if IDX_UNKNOWN_WORD in ngram
                else "index: " + str(ngram if len(ngram) > 1 else ngram[0])
            )
        )

        if IDX_UNKNOWN_WORD in ngram:
            print()
            return

        cat_len = max([
            len(self.get_category_name(ic))
            for ic in xrange(len(self.__categories__))
        ])
        cat_len = max(cat_len, 35)
        header_template = Print.style.bold(
            " {:<%d} |    fr    |  lv   |  sg   |  sn   |  gv   |  cv   |"
            % cat_len
        )
        print()
        print(header_template.format("Category"))
        header_template = (
            " {:-<%d}-|----------|-------|-------|-------|-------|-------|"
            % cat_len
        )
        print(header_template.format(''))
        row_template = (
            " %s | {:^8} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |"
            % (Print.style.warning("{:<%d}" % cat_len))
        )
        for icat in xrange(len(self.__categories__)):
            n_gram_tip = self.__trie_node__(ngram, icat)
            if n_gram_tip:
                print(
                    row_template
                    .format(
                        self.get_category_name(icat)[:35],
                        n_gram_tip[FR],
                        self.__lv__(ngram, icat),
                        self.__sg__(ngram, icat),
                        self.__sn__(ngram, icat),
                        self.__gv__(ngram, icat),
                        self.__cv__(ngram, icat),
                    )
                )
        print()

    def plot_value_distribution(self, cat):
        """
        绘制类别的全局和局部值分布。

        :param cat: the category name
        :type cat: str
        :raises: InvalidCategoryError
        """
        if self.get_category_index(cat) == IDX_UNKNOWN_CATEGORY:
            raise InvalidCategoryError

        import matplotlib.pyplot as plt

        icat = self.get_category_index(cat)
        vocab_metrics = self.__get_category_vocab__(icat)

        x = []
        y_lv = []
        y_gv = []
        vocab_metrics_len = len(vocab_metrics)

        for i in xrange(vocab_metrics_len):
            metric = vocab_metrics[i]
            x.append(i + 1)
            y_lv.append(metric[2])
            y_gv.append(metric[3])

        plt.figure(figsize=(20, 10))
        plt.title(
            "Word Value Distribution (%s)" % self.get_category_name(icat)
        )

        plt.xlabel("Word Rank")
        plt.ylabel("Value")
        plt.xlim(right=max(x))

        plt.plot(
            x, y_lv, "-", label="local value ($lv$)",
            linewidth=2, color="#7f7d7e"
        )
        plt.plot(
            x, y_gv, "-", label="global value ($gv$)",
            linewidth=4, color="#2ca02c")
        plt.legend()

        plt.show()

    def extract_insight(
        self, doc, cat='auto', level='word', window_size=3, min_cv=0.01, sort=True
    ):
        """
        获取分类决策中涉及的文本块列表。
        给定一个文档，返回分类决策，以及相关的置信值。如果给定了类别，则执行该过程，如果给定的类别是分类器指定的类别。
        :param doc: 文档的内容）
        :type doc: str
        :param cat: 获取文本块的类别。如果不存在，它将自动使用分配的类别分类后通过SS3。
        :type cat: str
        :param level: 要提取文本块的级别。选项包括“单词”、“句子”或“段落”。（默认值：'word'））
        :type level: str
        :param window_size: （每个标识单词前后的单词数，也包括在识别的单词中。例如，``window_size=0``表示只返回单个单词，
                            ``window_size=1``表示还包括在他们之前和之后。如果选择了多个单词足够接近，单词窗口重叠，
                            然后，这些单词窗口将被合并为一个更长的单一窗口。当“level”不等于“word”时，将忽略此参数。（默认值：3））
        :type window_size: int
        :param min_cv:每个文本块必须具有的最小置信值包含在输出中。（默认值0.01））
        :type min_cv: float
        :param sort: 是否返回按置信值排序的文本块或者没有。如果``sort=False``，则将返回块按照输入文档中的顺序。（默认值：True））
        :type sort: bool
        :returns: 包含相关文本（块）的成对（文本、置信值）列表，以及分类决策的程度（*）。（*）由置信值给出）
        :rtype: list
        :raises: InvalidCategoryError, ValueError
        """
        r = self.classify(doc, json=True)
        word_regex = self.__word_regex__

        if cat == 'auto':
            c_i = r["cvns"][0][0]
        else:
            c_i = self.get_category_index(cat)
            if c_i == IDX_UNKNOWN_CATEGORY:
                Print.error(
                    "The excepted values for the `cat` argument are 'auto' "
                    "or a valid category name, found '%s' instead" % str(cat),
                    raises=InvalidCategoryError
                )

        if level == 'paragraph':
            insights = [
                (
                    "".join([word["lexeme"]
                             for s in p["sents"]
                             for word in s["words"]]),
                    p["cv"][c_i]
                )
                for p in r["pars"]
                if p["cv"][c_i] > min_cv
            ]
        elif level == 'sentence':
            insights = [
                (
                    "".join([word["lexeme"]
                             for word in s["words"]]),
                    s["cv"][c_i]
                )
                for p in r["pars"] for s in p["sents"]
                if s["cv"][c_i] > min_cv
            ]
        elif level == 'word':
            ww_size = window_size
            insights = []
            for p in r["pars"]:
                words = [w for s in p["sents"] for w in s["words"]]
                w_i = 0
                while w_i < len(words):
                    w = words[w_i]
                    if w["cv"][c_i] > min_cv:
                        ww = []
                        ww_cv = 0
                        ww_left = min(w_i, ww_size) + 1
                        w_i -= ww_left - 1
                        while ww_left > 0 and w_i < len(words):

                            ww.append(words[w_i]["lexeme"])
                            ww_cv += words[w_i]["cv"][c_i]

                            if words[w_i]["cv"][c_i] > min_cv:
                                ww_left += min(ww_size, (len(words) - 1) - w_i)

                            if re.search(word_regex, words[w_i]["lexeme"]):
                                ww_left -= 1

                            w_i += 1

                        insights.append(("".join(ww), ww_cv))
                    else:
                        w_i += 1
        else:
            raise ValueError(
                "expected values for the `level` argument are "
                "'word', 'sentence', or 'paragraph', found '%s' instead."
                % str(level)
            )

        if sort:
            insights.sort(key=lambda b_cv: -b_cv[1])
        return insights

    def learn(self, doc, cat, n_grams=1, prep=True, update=True):
        """
       学习给定类别的新文档。

        :param doc: 文档目录
        :type doc: str
        :param cat: 类别名称
        :type cat: str
        :type n_grams: int
        :param prep: 开启默认的输入预处理
        :type prep: bool
        :param update: 学习之后模型自动更新
        :type update: bool
        """
        self.__cv_cache__ = None

        if not doc or cat is None:
            return

        try:
            doc = doc.decode(ENCODING)
        except UnicodeEncodeError:  # for python 2 compatibility
            doc = doc.encode(ENCODING).decode(ENCODING)
        except AttributeError:
            pass

        icat = self.__get_category__(cat)
        cat = self.__categories__[icat]
        word_to_index = self.__word_to_index__
        word_regex = self.__word_regex__

        if prep:
            Print.info("preprocessing document...", offset=1)
            stime = time()
            doc = Pp.clean_and_ready(doc)
            Print.info("finished --time: %.1fs" % (time() - stime), offset=1)
        doc = re.findall("%s|[^%s]+" % (word_regex, self.__word_delimiter__), doc)

        text_len = len(doc)
        Print.info(
            "about to learn new document (%d terms)" % text_len, offset=1
        )

        vocab = cat[VOCAB]  # getting cat vocab

        index_to_word = self.__index_to_word__
        max_frs = self.__max_fr__[icat]
        max_gvs = self.__max_gv__[icat]

        stime = time()
        Print.info("learning...", offset=1)
        tips = []
        for word in doc:
            if re.match(word_regex, word):
                self.__prun_counter__ += 1
                # if word doesn't exist yet, then...
                try:
                    word = word_to_index[word]
                except KeyError:
                    new_index = len(word_to_index)
                    word_to_index[word] = new_index
                    index_to_word[new_index] = word
                    word = new_index

                tips.append(vocab)

                if len(tips) > n_grams:
                    del tips[0]

                tips_length = len(tips)

                for i in xrange(tips_length):
                    tips_i = tips[i]

                    try:
                        max_frs[i]
                    except IndexError:
                        max_frs.append(1)
                        max_gvs.append(0)

                    try:
                        word_info = tips_i[word]
                        word_info[FR] += 1

                        if word_info[FR] > max_frs[(tips_length - 1) - i]:
                            max_frs[(tips_length - 1) - i] = word_info[FR]
                    except KeyError:
                        tips_i[word] = [
                            {},  # NEXT/VOCAB
                            1,   # FR
                            0,   # CV
                            0,   # SG
                            0,   # GV
                            0    # LV
                        ]
                        word_info = tips_i[word]

                    # print i, index_to_word[ word ], tips_i[word][FR]
                    tips[i] = word_info[NEXT]
            else:
                tips[:] = []
                if self.__prun_counter__ >= self.__prun_trigger__:
                    # trie data-structures pruning
                    self.__prune_tries__()

        Print.info("finished --time: %.1fs" % (time() - stime), offset=1)
        # updating values
        if update:
            self.update_values(force=True)

    def classify(self, doc, prep=True, sort=True, json=False, prep_func=None):
        """
       对给定文档进行分类。

        :param doc: 文档内容
        :type doc: str
        :param prep: 启用默认输入预处理（默认：True）
        :type prep: bool
        :param sort: 对分类结果进行排序
        :type sort: bool
        :param json: 以json格式返回结果的调试版本。
        :type json: bool
        :param prep_func:要应用的自定义预处理函数对给定文档进行分类之前。如果未给出，默认预处理函数将被使用（只要``prep=True``））
        :type prep_func: function
        :returns: 如果``sort``为False，则返回文档置信向量。如果``sort``为True，则为配对列表（类别指数，置信值）按置信值排序。）
        :rtype: list
        :raises: EmptyModelError
        """
        if not self.__categories__:
            raise EmptyModelError

        if self.__update_needed__():
            self.update_values()

        doc = doc or ''
        try:
            doc = doc.decode(ENCODING)
        except UnicodeEncodeError:  # for python 2 compatibility
            doc = doc.encode(ENCODING).decode(ENCODING)
        except BaseException:
            pass

        if not json:
            paragraphs_cvs = [
                self.__classify_paragraph__(parag, prep=prep, prep_func=prep_func)
                for parag in re.split(self.__parag_delimiter__, doc)
                if parag
            ]
            if paragraphs_cvs:
                cv = self.summary_op_paragraphs(paragraphs_cvs)

            else:
                cv = self.__zero_cv__

            if sort:

                y=sorted(
                    [
                        (i, cv[i])
                        for i in xrange(len(cv))
                    ],
                    key=lambda e: -e[1]
                )

                '''
                进行劣势数据进行置信度增强
                '''

                lst=list(y[0])
                lst2=list(y[1])
                lst2[1]=1.03*lst2[1]
                if lst[1]<lst2[1]:#判断是否增强有效
                    m_param=lst[0]
                    lst[0]=lst2[0]
                    lst2[0]=m_param
                t=tuple(lst)
                t2=tuple(lst2)
                y.pop()#删除原元组
                y.pop()
                y.append(t)#新增增强元组
                y.append(t2)



                '''end'''
                return y
                '''
                return sorted(
                    [
                        (i, cv[i])
                        for i in xrange(len(cv))
                    ],
                    key=lambda e: -e[1]
                )
                '''

                '''cv改动（进行阈值偏移）'''


                '''end'''

            return cv
        else:
            info = [
                self.__classify_paragraph__(parag, prep=prep, prep_func=prep_func, json=True)
                for parag in re_split_keep(self.__parag_delimiter__, doc)
                if parag
            ]

            nbr_cats = len(self.__categories__)
            cv = self.summary_op_paragraphs([v["cv"] for v in info])
            max_v = max(cv)

            if max_v > 1:
                norm_cv = map(lambda x: x / max_v, cv)
            else:
                norm_cv = cv


            norm_cv_sorted = sorted(
                [(i, nv, cv[i]) for i, nv in enumerate(norm_cv)],
                key=lambda e: -e[1]
            )


            return {
                "pars": info,
                "cv": cv,
                "wmv": reduce(vmax, [v["wmv"] for v in info]),
                "cvns": norm_cv_sorted,
                "ci": [self.get_category_name(ic) for ic in xrange(nbr_cats)]
            }

    def classify_label(self, doc, def_cat=STR_MOST_PROBABLE, labels=True, prep=True):
        """
        对返回类别标签的给定文档进行分类。

        :param doc: 文档内容
        :type doc: str
        :param def_cat: SS3未指定时要分配的默认类别能够对文档进行分类。选项包括“最有可能”、“未知”或给定的类别名称。（默认值：“最有可能”））
        :type def_cat: str
        :param labels: 是返回类别标签还是仅返回类别索引（默认值：True））
        :type labels: bool
        :param prep:启用默认输入预处理过程（默认：True）
        :type prep: bool
        :returns: 类别标签或类别索引。
        :rtype: str or int
        :raises: InvalidCategoryError
        """
        r = self.classify(doc, sort=True, prep=prep)

        if not r or not r[0][1]:
            if not def_cat or def_cat == STR_UNKNOWN:
                cat = STR_UNKNOWN_CATEGORY


            elif def_cat == STR_MOST_PROBABLE:
                cat = self.get_most_probable_category()


            else:
                if self.get_category_index(def_cat) == IDX_UNKNOWN_CATEGORY:
                    raise InvalidCategoryError

                cat = def_cat

        else:
            cat = self.get_category_name(r[0][0])



        return cat if labels else self.get_category_index(cat)

    def classify_multilabel(self, doc, def_cat=STR_UNKNOWN, labels=True, prep=True):
        """
       对返回多个类别标签的给定文档进行分类。
        此方法可用于执行多标签分类。在内部，它使用置信向量上的k-mean聚类来选择适当的标签。
        """

        r = self.classify(doc, sort=True, prep=prep)

        if not r or not r[0][1]:
            if not def_cat or def_cat == STR_UNKNOWN:
                return []
            elif def_cat == STR_MOST_PROBABLE:
                cat = self.get_most_probable_category()
            else:
                if self.get_category_index(def_cat) == IDX_UNKNOWN_CATEGORY:
                    raise InvalidCategoryError
                cat = def_cat
            if cat != STR_OTHERS_CATEGORY:
                return [cat] if labels else [self.get_category_index(cat)]
            else:
                return []
        else:
            __other_idx__ = self.get_category_index(STR_OTHERS_CATEGORY)
            if labels:
                result = [self.get_category_name(cat_i)
                          for cat_i, _ in r[:kmean_multilabel_size(r)]]
                # removing "hidden" special category ("[other]")
                if __other_idx__ != IDX_UNKNOWN_CATEGORY and STR_OTHERS_CATEGORY in result:
                    result.remove(STR_OTHERS_CATEGORY)
            else:
                result = [cat_i for cat_i, _ in r[:kmean_multilabel_size(r)]]
                # removing "hidden" special category ("[other]")
                if __other_idx__ != IDX_UNKNOWN_CATEGORY and __other_idx__ in result:
                    result.remove(__other_idx__)
            return result


    def fit(self, x_train, y_train, n_grams=1, prep=True, leave_pbar=True):
        """
        给模型一份文档和类别标签列表，对其进行训练。

        :param x_train: 文档列表
        :type x_train: list (of str)
        :param y_train: 文档标签列表
        :type y_train: 用于单标签分类的str列表；多标签分类的str列表列表。
        :param n_grams: 指示要学习的最大``n``-grams 例如，值``1``仅表示1-gram（单词），``2``表示1和2-gram，``3``、1-gram、2-gram和3-gram，依此类推。
        :type n_grams: int
        :param prep: 默认的输入预处理
        :type prep: bool
        :param leave_pbar: 控制是否离开进度条，完成后将其移除。
        :type leave_pbar: bool
        :raises: ValueError
        """
        stime = time()
        x_train, y_train = list(x_train), list(y_train)

        if len(x_train) != len(y_train):
            raise ValueError("`x_train` and `y_train` must have the same length")

        if len(y_train) == 0:
            raise ValueError("`x_train` and `y_train` are empty")

        # 如果是多标签分类问题
        if is_a_collection(y_train[0]):
            # flattening y_train
            labels = [l for y in y_train for l in y]  # 多标签列表
            self.__multilabel__ = True  # 点亮多标签标志位
        else:
            labels = y_train  # 单标签列表

        cats = sorted(list(set(labels)))

        # 如果是单标签分类问题
        if not is_a_collection(y_train[0]):
            """添加单词到类别字典中"""
            x_train = [
                "".join([
                    x_train[i]
                    if x_train[i] and x_train[i][-1] == '\n'
                    else
                    x_train[i] + '\n'
                    for i in xrange(len(x_train))
                    if y_train[i] == cat
                ])
                for cat in cats
            ]
            y_train = list(cats)

        Print.info("about to start training", offset=1)
        Print.verbosity_region_begin(VERBOSITY.NORMAL)
        progress_bar = tqdm(total=len(x_train), desc="Training",
                            leave=leave_pbar, disable=Print.is_quiet())

        # 如果是多标签分类问题
        if is_a_collection(y_train[0]):
            __others__ = [STR_OTHERS_CATEGORY]
            for i in range(len(x_train)):
                for label in (y_train[i] if y_train[i] else __others__):
                    self.learn(
                        x_train[i], label,
                        n_grams=n_grams, prep=prep, update=False
                    )
                progress_bar.update(1)
        else:
            for i in range(len(x_train)):
                progress_bar.set_description_str("Training on '%s'" % str(y_train[i]))
                self.learn(
                    x_train[i], y_train[i],
                    n_grams=n_grams, prep=prep, update=False
                )
                progress_bar.update(1)
        progress_bar.close()
        self.__prune_tries__()
        Print.verbosity_region_end()
        Print.info("finished --time: %.1fs" % (time() - stime), offset=1)
        self.update_values(force=True)  # 更新全局置信度

    def predict_proba(self, x_test, prep=True, leave_pbar=True):
        """
       对返回置信向量列表的文档列表进行分类。

        :param x_test: 要分类的文档列表
        :type x_test: list (of str)
        :param prep: 启用默认输入预处理（默认：True）
        :type prep: bool
        :param leave_pbar: c控制是否离开进度条，完成后将其移除。
        :type leave_pbar: bool
        :returns: 置信向量列表
        :rtype: list (of list of float)
        :raises: EmptyModelError
        """
        if not self.__categories__:
            raise EmptyModelError

        if self.get_ngrams_length() == 1 and self.__summary_ops_are_pristine__():
            return self.__predict_fast__(x_test, prep=prep,
                                         leave_pbar=leave_pbar, proba=True)

        x_test = list(x_test)
        classify = self.classify
        return [
            classify(x, sort=False)
            for x in tqdm(x_test, desc="Classification", disable=Print.is_quiet())
        ]

    def predict(
        self, x_test, def_cat=None,
        labels=True, multilabel=False, prep=True, leave_pbar=True
    ):
        """
       对文档列表进行分类。

        :param x_test: 要分类的文档列表
        :type x_test: list (of str)
        :param def_cat: SS3未指定时要分配的默认类别能够对文档进行分类。选项包括“最有可能”、“未知”或给定的类别名称。
                         （默认值：“最有可能”，或“未知”多标签分类））
        :type def_cat: str
        :param labels: 是返回类别名称列表还是仅返回类别索引）
        :type labels: bool
        :param multilabel: 是否执行多标签分类。如果启用，对于每个文档，将返回标签的“列表”而不是单个标签（``str``）。
                           如果模型是使用多标签数据训练的，那么参数将被忽略并设置为True。）
        :type multilabel: bool
        :param leave_pbar: 控制是否离开进度条，完成后将其移除。
        :type leave_pbar: bool
        :returns: 如果标签正确，为类别名称列表，否则为类别索引列表
        :rtype: list (of int or str)
        :raises: EmptyModelError, InvalidCategoryError
        """
        if not self.__categories__:
            raise EmptyModelError

        multilabel = multilabel or self.__multilabel__

        if def_cat is None:
            def_cat = STR_UNKNOWN if multilabel else STR_MOST_PROBABLE

        if not def_cat or def_cat == STR_UNKNOWN:
            if not multilabel:
                Print.info(
                    "default category was set to 'unknown' (its index will be -1)",
                    offset=1
                )
        else:
            if def_cat == STR_MOST_PROBABLE:
                Print.info(
                    "default category was automatically set to '%s' "
                    "(the most probable one)" % self.get_most_probable_category(),
                    offset=1
                )
            else:
                Print.info("default category was set to '%s'" % def_cat, offset=1)
                if self.get_category_index(def_cat) == IDX_UNKNOWN_CATEGORY:
                    raise InvalidCategoryError

        if self.get_ngrams_length() == 1 and self.__summary_ops_are_pristine__():
            return self.__predict_fast__(x_test, def_cat=def_cat, labels=labels,
                                         multilabel=multilabel, prep=prep,
                                         leave_pbar=leave_pbar)

        stime = time()
        Print.info("about to start classifying test documents", offset=1)
        classify = self.classify_label if not multilabel else self.classify_multilabel
        x_test = list(x_test)

        y_pred = [
            classify(doc, def_cat=def_cat, labels=labels, prep=prep)
            for doc in tqdm(x_test, desc="Classification",
                           leave=leave_pbar, disable=Print.is_quiet())
        ]

        Print.info("finished --time: %.1fs" % (time() - stime), offset=1)
        return y_pred

    def cv(self, ngram, cat):
        """
        返回给定类别的给定单词n-gram的“置信值”。

        该值是通过对全局值进行最终转换而获得的使用gv函数[*]计算给定单词n-gram的值。

        这些转换是在创建新的SS3实例时给出的（请参阅SS3类构造函数的``cv_m``参数以了解更多信息）。

        Examples:

            >>> clf.cv("chicken", "food")
            >>> clf.cv("roast chicken", "food")
            >>> clf.cv("chicken", "sports")

        :param ngram: 单词或单词n-gram）
        :type ngram: str
        :param cat:类别标签
        :type cat: str
        :returns: 置信度
        :rtype: float
        :raises: InvalidCategoryError
        """
        return self.__apply_fn__(self.__cv__, ngram, cat)

    def gv(self, ngram, cat):
        """
       返回给定类别的给定单词n-gram的“全局值”。

        Examples:

            >>> clf.gv("chicken", "food")
            >>> clf.gv("roast chicken", "food")
            >>> clf.gv("chicken", "sports")

        """
        return self.__apply_fn__(self.__gv__, ngram, cat)

    def lv(self, ngram, cat):
        """
       返回给定类别的给定单词n-gram的“局部值”。

        Examples:

            >>> clf.lv("chicken", "food")
            >>> clf.lv("roast chicken", "food")
            >>> clf.lv("chicken", "sports")

        """
        return self.__apply_fn__(self.__lv__, ngram, cat)

    def sg(self, ngram, cat):
        """
        返回给定类别的给定单词n-gram的“显著性因子”。）

        Examples:

            >>> clf.sg("chicken", "food")
            >>> clf.sg("roast chicken", "food")
            >>> clf.sg("chicken", "sports")

        :param cat: 类别标签
        :type cat: str
        :returns: 显著性因子
        :rtype: float
        :raises: InvalidCategoryError
        """
        return self.__apply_fn__(self.__sg__, ngram, cat)

    def sn(self, ngram, cat):
        """
        返回给定类别的给定单词n-gram的“制裁因子”。）


        Examples:

            >>> clf.sn("chicken", "food")
            >>> clf.sn("roast chicken", "food")
            >>> clf.sn("chicken", "sports")


        :returns: 制裁因子）
        :rtype: float
        :raises: InvalidCategoryError
        """
        return self.__apply_fn__(self.__sn__, ngram, cat)


class SS3Vectorizer(CountVectorizer):
    r"""
    将文本文档集合转换为使用SS3模型加权的文档术语矩阵。文档d中术语t相对于类别c的权重通过乘以术语频率权重（tf_weight）和基于SS3的权重（SS3_weight
    """

    __clf__ = None
    __icat__ = None
    __ss3_weight__ = None
    __tf_weight__ = None

    def __init__(self, clf, cat, ss3_weight='only_cat', tf_weight='raw_count', top_n=None,
                 **kwargs):

        if clf.get_category_index(cat) == IDX_UNKNOWN_CATEGORY:
            raise InvalidCategoryError

        if not callable(ss3_weight) and ss3_weight not in WEIGHT_SCHEMES_SS3:
            raise ValueError("`ss3_weight` argument must be either a custom "
                             "function or any of the following strings: %s" %
                             ", ".join(WEIGHT_SCHEMES_SS3))
        if not callable(tf_weight) and tf_weight not in WEIGHT_SCHEMES_TF:
            raise ValueError("`tf_weight` argument must be either a custom "
                             "function or any of the following strings: %s" %
                             ", ".join(WEIGHT_SCHEMES_TF))

        if top_n is not None:
            if not isinstance(top_n, numbers.Integral) or top_n <= 0:
                raise ValueError("`top_n` argument must be either a positive integer or None")

        ss3_n_grams = clf.get_ngrams_length()
        min_n, max_n = kwargs["ngram_range"] if "ngram_range" in kwargs else (1, 1)
        if not isinstance(min_n, numbers.Integral) or (
           not isinstance(max_n, numbers.Integral)) or (min_n > max_n or min_n <= 0):
            raise ValueError("`ngram_range` (n0, n1) argument must be a valid n-gram range "
                             "where n0 and n1 are positive integer such that n0 <= n1.")
        if max_n > ss3_n_grams:
            Print.warn("`ngram_range` (n0, n1) argument, n1 is greater than the longest n-gram "
                       "learned by the given SS3 model")
            min_n, max_n = min(min_n, ss3_n_grams), min(max_n, ss3_n_grams)

        if "dtype" not in kwargs:
            kwargs["dtype"] = float

        self.__clf__ = clf
        self.__icat__ = clf.get_category_index(cat)

        if ss3_weight == WEIGHT_SCHEMES_SS3[0]:    # 'only_cat'
            self.__ss3_weight__ = lambda cv, icat: cv[icat]
        elif ss3_weight == WEIGHT_SCHEMES_SS3[1]:  # 'diff_all'
            self.__ss3_weight__ = lambda cv, icat: cv[icat] - sum([cv[i]
                                                                  for i in range(len(cv))
                                                                  if i != icat])
        elif ss3_weight == WEIGHT_SCHEMES_SS3[2]:  # 'diff_max'
            self.__ss3_weight__ = lambda cv, icat: cv[icat] - max([cv[i]
                                                                   for i in range(len(cv))
                                                                   if i != icat])
        elif ss3_weight == WEIGHT_SCHEMES_SS3[3]:  # 'diff_median'
            self.__ss3_weight__ = lambda cv, icat: cv[icat] - sorted(cv)[
                len(cv) // 2 - int(not (len(cv) % 2))
            ]
        elif ss3_weight == WEIGHT_SCHEMES_SS3[4]:  # 'diff_mean'
            self.__ss3_weight__ = lambda cv, icat: cv[icat] - sum(cv) / float(len(cv))
        else:
            self.__ss3_weight__ = ss3_weight

        if "binary" in kwargs and kwargs["binary"]:
            tf_weight = "binary"
            del kwargs["binary"]

        if tf_weight in WEIGHT_SCHEMES_TF[:2]:    # 'binary' or 'raw_count'
            self.__tf_weight__ = lambda freqs, iterm: freqs[iterm]
        elif tf_weight == WEIGHT_SCHEMES_TF[2]:  # 'term_freq'
            self.__tf_weight__ = lambda freqs, iterm: freqs[iterm] / np.sum(freqs)
        elif tf_weight == WEIGHT_SCHEMES_TF[3]:  # 'log_norm'
            self.__tf_weight__ = lambda freqs, iterm: log(1 + freqs[iterm])
        elif tf_weight == WEIGHT_SCHEMES_TF[4]:  # 'double_norm'
            self.__tf_weight__ = lambda freqs, iterm: .5 + .5 * freqs[iterm] / np.max(freqs)
        else:
            self.__tf_weight__ = tf_weight

        if "vocabulary" in kwargs:
            vocabulary = kwargs["vocabulary"]
            del kwargs["vocabulary"]
        else:
            icat = self.__icat__
            vocabularies_out = [[] for _ in range(max_n)]
            clf.__get_vocabularies__(icat, clf.__categories__[icat][VOCAB],
                                     [], max_n, vocabularies_out, " ")
            vocabulary = []
            for i_gram in range(min_n - 1, max_n):
                vocabulary += [t[0]
                               for t
                               in sorted(vocabularies_out[i_gram], key=lambda k: -k[-1])[:top_n]]

        super().__init__(binary=(tf_weight == "binary"), vocabulary=vocabulary, **kwargs)

    def fit_transform(self, raw_documents):
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        dtm = super().transform(raw_documents)

        # caching in-loop variables
        clf = self.__clf__
        ss3_weight = self.__ss3_weight__
        tf_weight = self.__tf_weight__
        icat = self.__icat__
        clf_apply = clf.__apply_fn__
        clf_cv = clf.__classify_ngram__
        feature_names = self.get_feature_names()
        indptr, indices, data = dtm.indptr, dtm.indices, dtm.data

        for i_row in range(dtm.shape[0]):
            doc_freqs = data[indptr[i_row]:indptr[i_row + 1]].copy()
            for offset in range(indptr[i_row + 1] - indptr[i_row]):
                i_col = indptr[i_row] + offset
                term = feature_names[indices[i_col]]
                term_cv = clf_apply(clf_cv, term, None)

                data[i_col] = tf_weight(doc_freqs, i_col) * ss3_weight(term_cv, icat)

        return dtm  # document-term matrix


class EmptyModelError(Exception):
    """模型为空时引发异常"""

    def __init__(self, msg=''):
        """类构造函数"""
        Exception.__init__(
            self,
            "The model is empty (it hasn't been trained yet)."
        )


class InvalidCategoryError(Exception):
    """类别无效时引发的异常"""

    def __init__(self, msg=''):
        """Class constructor."""
        Exception.__init__(
            self,
            "The given category is not valid"
        )


def kmean_multilabel_size(res):
    """
    使用k-means指示在何处分割“SS3.classify”的输出。
    给定一个``SS3.classify``的输出（``res``），告诉它在哪里分区，分为2个集群，以便其中一个集群包含的类别标签，分类器在执行多标签分类时应该输出。
    为此，在``res``中的类别置信值。
    :param res: SS3.classify的分类输出
    :type res:列表（排序对（类别、置信度值））
    :returns: 一个正整数，指示分割‘res’的位置
    :rtype: int
    """
    cent = {"neg": -1, "pos": -1}  # centroids (2 clusters: "pos" and "neg")
    clust = {"neg": [], "pos": []}  # clusters (2 clusters: "pos" and "neg")
    new_cent_neg = res[-1][1]
    new_cent_pos = res[0][1]

    if new_cent_neg == new_cent_pos:
        return 0

    while (cent["pos"] != new_cent_pos) or (cent["neg"] != new_cent_neg):
        cent["neg"], cent["pos"] = new_cent_neg, new_cent_pos
        clust["neg"], clust["pos"] = [], []
        for _, cat_cv in res:
            if abs(cent["neg"] - cat_cv) < abs(cent["pos"] - cat_cv):
                clust["neg"].append(cat_cv)
            else:
                clust["pos"].append(cat_cv)
        if len(clust["neg"]) > 0:
            new_cent_neg = sum(clust["neg"]) / len(clust["neg"])
        if len(clust["pos"]) > 0:
            new_cent_pos = sum(clust["pos"]) / len(clust["pos"])
    return len(clust["pos"])


def sigmoid(v, l):
    """sigmoid函数"""
    try:
        return .5 * tanh((3. / l) * v - 3) + .5
    except ZeroDivisionError:
        return 0


def mad(values, n):
    """中位数绝对偏差平均值。"""
    if len(values) < n:
        values += [0] * int(n - len(values))
    values.sort()
    if n == 2:
        return (values[0], values[0])
    values_m = n // 2 if n % 2 else n // 2 - 1
    m = values[values_m]  # Median
    sd = sum([abs(m - lv) for lv in values]) / float(n)  # sd Mean
    return m, sd


def key_as_int(dct):
    """将给定的字典（数字）键转换为int。"""
    keys = list(dct)
    if len(keys) and keys[0].isdigit():
        new_dct = {}
        for key in dct:
            new_dct[int(key)] = dct[key]
        return new_dct
    return dct


def re_split_keep(regex, string):
    """
    通过re.split强制包含不匹配的项。
    这允许在分割输入后保留原始内容供以后使用的文档（例如，用于从现场测试中使用）
    """
    if not re.match(r"\(.*\)", regex):
        regex = "(%s)" % regex
    return re.split(regex, string)


def list_hash(str_list):
    """
    返回给定字符串列表的哈希值。

    :param str_list:  （字符串列表）(e.g. x_test)
    :type str_list: list (of str)
    :returns: MD5哈希值
    :rtype: str
    """
    import hashlib
    m = hashlib.md5()
    for doc in str_list:
        try:
            m.update(doc)
        except (TypeError, UnicodeEncodeError):
            m.update(doc.encode('ascii', 'ignore'))
    return m.hexdigest()


def vsum(v0, v1):
    """矢量求和。"""
    return [v0[i] + v1[i] for i in xrange(len(v0))]


def vmax(v0, v1):
    """最大矢量"""
    return [max(v0[i], v1[i]) for i in xrange(len(v0))]


def vdiv(v0, v1):
    """矢量除法"""
    return [v0[i] / v1[i] if v1[i] else 0 for i in xrange(len(v0))]


def set_verbosity(level):
    """
    设置详细级别。
        - ``0`` (quiet): （安静）：不输出任何消息（仅错误消息）
        - ``1`` (normal): （正常）：默认行为，仅显示警告消息和进度条
        - ``2`` (verbose):（verbose）：还显示信息性非基本消息

        以下内置常量也可用于引用这3个值：
        ``语言流畅。安静``，``口若悬河。正常``和``措辞。分别是VERBOSE ``。

        例如，如果您希望PySS3隐藏所有内容，甚至进度条，您可以简单地执行以下操作：）
    >>> import pyss3
    ...
    >>> pyss3.set_verbosity(0)
    ...
    >>> # here's the rest of your code :D

    or, equivalently:

    >>> import pyss3
    >>> from pyss3 import VERBOSITY
    ...
    >>> pyss3.set_verbosity(VERBOSITY.QUIET)
    ...
    >>> # here's the rest of your code :D

    :param level: the verbosity level
    :type level: int
    """
    Print.set_verbosity(level)


# user-friendly aliases
SS3.set_smoothness = SS3.set_s
SS3.get_smoothness = SS3.get_s
SS3.set_significance = SS3.set_l
SS3.get_significance = SS3.get_l
SS3.set_sanction = SS3.set_p
SS3.get_sanction = SS3.get_p
SS3.set_alpha = SS3.set_a
SS3.get_alpha = SS3.get_a
SS3.get_alpha = SS3.get_a
SS3.train = SS3.fit
SS3.save = SS3.save_model
SS3.load = SS3.load_model
SS3.update = SS3.update_values
