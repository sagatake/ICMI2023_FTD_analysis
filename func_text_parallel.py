#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:48:23 2021
Updated on Mon Mar 6 21:43:00 2023

@author: takeshi-s
"""
from matplotlib import pyplot as plt
import pprint as pp
import pandas as pd
import numpy as np
import traceback
import shutil
import math
import time
import csv
import sys
import os

from sudachipy import tokenizer
from sudachipy import dictionary
from scipy.spatial.distance import cosine


import torch

# As of Nov 16 2021, for installation, please follow https://qiita.com/m__k/items/863013dbe847dc613844 
from transformers import BertModel, BertForNextSentencePrediction
#from transformers.tokenization_bert_japanese import BertJapaneseTokenizer as BertTokenizer
from transformers import BertJapaneseTokenizer as BertTokenizer

import difflib

#To run following function, need to insatall MeCab and the latest neologdn dictionary files
#import MeCab
#import neologdn
#wakati = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
#tagger = MeCab.Tagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

#As of Nov 16 2021, transformers(cl-tohoku) uses fugashi for tokenization
#To install fugashi, https://pypi.org/project/fugashi/
from fugashi import Tagger, GenericTagger
from sudachipy import tokenizer, dictionary
from  pyknp import Juman, BList, KNP

DEBUG = None

class Func_text(object):
    
    def __init__(self, WITH_BERT = True):
        
        #tagger = Tagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        self.tagger = Tagger()
        #tagger = GenericTagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        #tagger = GenericTagger()
        
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C
        
        self.knp = KNP(option="-tab -anaphora", multithreading=True)
        
        #dummy value. This will be alwayse over-rided in the parent source code.
        
        if WITH_BERT:
            self.MAX_TOKEN_LEN = 512
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print('Loading BERT ...')
            self.bert_tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
            self.bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
            self.bert_model.to(self.device)
            print('Done')
    
    
    def calc_CDI_J(self, text, negation_weight = 1):
        #Categorical Dynamic Index, calculated from PCA loadings from 1st component with function words (+negation)
        #
        # ['num_negation' 'ratio_接頭辞' 'ratio_感動詞' 'ratio_接尾辞' 'ratio_代名詞'
        #  'ratio_連体詞' 'ratio_接続詞' 'ratio_助詞' 'ratio_助動詞']
        # explained variance ratio, component 0,    0.1749346852
        # ##### component load #####
        # array([[-0.4642981 ,  0.14165819, -0.23058397,  0.09469763, -0.13429847,
        #         -0.01211669, -0.10448369, -0.6553614 ,  0.4935844 ]],
        #       dtype=float32)

        pos_dict        = self.count_POS_sudachi(text)
        num_negation    = self.count_negation(text)
        
        sum_freq = sum([pos_dict[key] for key in pos_dict.keys()])
        for key in pos_dict.keys():
            pos_dict[key] = pos_dict[key] / sum_freq
        
        CDI = - num_negation + pos_dict['接頭辞'] - pos_dict['感動詞'] + pos_dict['接尾辞'] - pos_dict['代名詞']
        - pos_dict['連体詞'] - pos_dict['接続詞'] - pos_dict['助詞'] + pos_dict['助動詞']
        
        return [CDI]

    def calc_CDI(self, text, negation_weight = 1):
        #Categorical Dynamic Index, associated Japanese POS with English POS
        
        pos_dict        = self.count_POS_sudachi(text)
        num_negation    = self.count_negation(text)
        
        sum_freq = sum([pos_dict[key] for key in pos_dict.keys()])
        for key in pos_dict.keys():
            pos_dict[key] = pos_dict[key] / sum_freq
        
        CDI = pos_dict['連体詞'] + pos_dict['助詞'] - pos_dict['代名詞'] 
        - (pos_dict['助動詞'] + pos_dict['接尾辞']) - pos_dict['接続詞']
        - negation_weight * num_negation - pos_dict['副詞']
        
        return [CDI]
    
    def count_POS_sudachi(self, text, 
                         tgt_pos = ['補助記号','空白','名詞','記号',
                                    '接頭辞','感動詞','副詞','接尾辞',
                                    '代名詞','形状詞','動詞','形容詞',
                                    '連体詞','接続詞',
                                    '助詞','助動詞'],
                         return_flags=False):
        
        tokenizer_obj = dictionary.Dictionary().create()
        mode = tokenizer_obj.SplitMode.C
        #out = tokenizer_obj.tokenize(text,mode)
        
        pos_dict = {}
        for pos in tgt_pos:
            pos_dict[pos] = 0
        
        for x in tokenizer_obj.tokenize(text, mode):
            if x.part_of_speech()[0] in tgt_pos:
                
                pos_dict[x.part_of_speech()[0]] += 1
    
        return pos_dict
    
    def count_negation(self, text):
        
        neg_count = 0
        
        for sentence in text:
            parse_result = self.knp.parse(sentence)
            for tag in parse_result.tag_list():
                # print('##############################')
                #pprint.pprint(dir(tag))
                fstring = tag.fstring
                units = fstring.split('<')[1:]
                units = [unit[:-1] for unit in units]
                for unit in units:
                    if '否定' in unit:
                        neg_count += 1
                        
        return neg_count
    
    def calc_WPM(self, text, audio_length):
        
        cnt = 0
        # frame_per_second = 30
        
        
        for IPU in text:
            tokens = self.tokenize_sudachi(IPU)
            cnt += len(tokens)
            
        # calc WPM
        feature = cnt / audio_length * 60
        
        return [feature]
    
    def count_backchannels(self, text):
        
        num_backchannel = 0
        
        for IPU in text:
            words =[m for m in self.tokenizer_obj.tokenize(IPU, self.mode)]
            if len(words) < 5:
                num_backchannel += 1
        
        return [num_backchannel]
    
    def BERT_sentence_average_diff(self, text, with_abs = True):
        
        # Tang et al., Natural language processing methods are sensitive to sub-clinical differences in schizophrenia spectrum
        # auther code: https://github.com/rekriz11/nlp_schizophrenia/blob/main/code/bert_random_walk.py
        
        BERT_embed_list = []
        for IPU in text:
            BERT_word_embed_list = self.get_BERT_embed(IPU)
            BERT_sentence_embed = np.average(BERT_word_embed_list, axis=0)
            BERT_embed_list.append(BERT_sentence_embed)
        score = self.calc_average_diff(BERT_embed_list, with_abs)
        
        return [score]
    
    def BERT_sentence_average_cosine(self, text):
        
        BERT_embed_list = []
        for IPU in text:
            BERT_word_embed_list = self.get_BERT_embed(IPU)
            BERT_sentence_embed = np.average(BERT_word_embed_list, axis=0)
            BERT_embed_list.append(BERT_sentence_embed)
        score = self.calc_average_cos(BERT_embed_list)
        
        return [score]
    
    def BERT_cont_word(self, text):
            
        tgt_pos = ['動詞', '形容詞', '名詞', '副詞', '形状詞']
        tgt_embeddings = []
        for IPU in text:
            
            BERT_tokens, BERT_ids, flag_list = self.match_BERT_sudachi_tokens(IPU,tgt_pos=tgt_pos)
            
            if len(BERT_tokens) > self.MAX_TOKEN_LEN:
                BERT_tokens = BERT_tokens[:self.MAX_TOKEN_LEN]
                BERT_ids = BERT_ids[:self.MAX_TOKEN_LEN]
            
            raw_embeddings = self.calc_BERT_embed(BERT_ids)
            for i in range(len(BERT_ids)):
                if flag_list[i]==1:
                    tgt_embeddings.append(raw_embeddings[i])
    
        score = self.calc_average_cos(tgt_embeddings)
        
        return [score]
    
    def BERT_word(self, text):
            
        tgt_pos = ['補助記号','空白','名詞','記号',
                    '接頭辞','感動詞','副詞','接尾辞',
                    '代名詞','形状詞','動詞','形容詞',
                    '連体詞','接続詞',
                    '助詞','助動詞']
        tgt_embeddings = []
        for IPU in text:
            
            BERT_tokens, BERT_ids, flag_list = self.match_BERT_sudachi_tokens(IPU,tgt_pos=tgt_pos)
            
            if len(BERT_tokens) > self.MAX_TOKEN_LEN:
                BERT_tokens = BERT_tokens[:self.MAX_TOKEN_LEN]
                BERT_ids = BERT_ids[:self.MAX_TOKEN_LEN]
            
            raw_embeddings = self.calc_BERT_embed(BERT_ids)
            for i in range(len(BERT_ids)):
                if flag_list[i]==1:
                    tgt_embeddings.append(raw_embeddings[i])
    
        score = self.calc_average_cos(tgt_embeddings)
        
        return [score]
    
    
    def calc_average_cos(self, embed_list):
        scores = []
        for i in range(0, len(embed_list)-1):
            #print(embed_list[i])
            #print(np.shape(embed_list[i+1]))
            scores.append(cosine(embed_list[i], embed_list[i+1]))
            #print(temp[-1])
            #input()
        score = np.average(scores)
        return score
    
    def calc_average_diff(self, embed_list, with_abs):
        scores = []
        for i in range(0, len(embed_list)-1):
            #print(embed_list[i])
            #print(np.shape(embed_list[i+1]))
            
            a = np.asarray(embed_list[i])
            b = np.asarray(embed_list[i+1])
            c = b - a
            
            if with_abs:
                diff = np.average(np.abs(c))
            else:
                diff = np.average(c)
            
            scores.append(diff)
            #print(temp[-1])
            #input()
        score = np.average(scores)
        return score
    
            
    def check_thanks(self, text):
            
        text = ''.join(text)
        flag = self._vocab_match(text, ['有り難う', 'サンキュー', 'どうも', '感謝'])
        
        return [flag]
    
    def count_content_words(self, text):
        
        num_total = 0
        num_content = 0
        #print(text)
        for IPU in text:
            words =[m for m in self.tokenizer_obj.tokenize(IPU, self.mode)]
            for word in words:
                
                dict_form = word.dictionary_form()
                norm_form = word.normalized_form()
                pos = word.part_of_speech()
                if DEBUG:
                    print(dict_form)
                    print(norm_form)
                    print(pos)
                
                if pos[0] in ['名詞','動詞', '形容詞', '副詞', '形状詞']:
                    #print(dict_form)
                    num_total += 1
                    num_content += 1
                
                else:
                    num_total += 1
                    
        feature = num_content/num_total
        
        return [feature]
    
    def count_punctuations(self, text):
        
        num_total = 0
        num_punct = 0
        #print(text)
        for IPU in text:
            words =[m for m in self.tokenizer_obj.tokenize(IPU, self.mode)]
            for word in words:
                
                dict_form = word.dictionary_form()
                norm_form = word.normalized_form()
                pos = word.part_of_speech()
                if DEBUG:
                    print(dict_form)
                    print(norm_form)
                    print(pos)
                
                if pos[0] in ['空白','補助記号']:
                    #print(dict_form)
                    num_total += 1
                    num_punct += 1
                
                else:
                    num_total += 1
                    
        feature = num_punct/num_total
        
        return [feature]
    
    def check_initial_que(self, text):
        
        flag = 0
        
        target_words = ['すみません', 'ねえ', 'ねえねえ']
        init_IPU = text[0]
        words = [m for m in self.tokenizer_obj.tokenize(init_IPU, self.mode)]
        for target_word in target_words:
            if target_word == words[0]:
                flag = 1
                break
        
        return [flag]
    
    def match_BERT_sudachi_tokens(self, raw_text, tgt_pos=['名詞']):
            extracted_tokens, sudachi_flags, raw_tokens = self.tokenize_sudachi(raw_text,
                                                    tgt_pos=tgt_pos,
                                                    return_flags=True)
            BERT_tokens, BERT_ids = self.tokenize_BERT(raw_text)
            
            matched_pairs = []
            
            output_flags = []
            j=0
            for i in range(1, len(BERT_tokens)-1):
                j_memory = j
                while True:
                    temp = BERT_tokens[i].replace('#','')
                    match_ratio = difflib.SequenceMatcher(None, temp, raw_tokens[j]).ratio()
                    #print(str(match_ratio) + ' : ' + BERT_tokens[i] + ' : ' + raw_tokens[j])
                    #input()
                    if match_ratio > 0.25:
                        if sudachi_flags[j]==1:
                            output_flags.append(1)
                        else:
                            output_flags.append(0)
    
                        matched_pairs.append([BERT_tokens[i], raw_tokens[j]])
                            
                        #since raw_tokens by sudach can be devided into several BERT_tokens, this process is needed
                        j-=3
                        if j<0:
                            j=0
                        
                        break
                    
                    j+=1
                    
                    #if no matched tokens were found
                    if j>=len(raw_tokens):
                        output_flags.append(999999999999999999999)
                        j = j_memory
                        break
                    
    
            #put flags for [CLS][SEP]
            output_flags.insert(0, 0)
            output_flags.append(0)
            
            if len(BERT_tokens)!=len(output_flags):
                print('No matched tokens were detected ...')
                sys.exit()
            
            #print('\n\n\n')           
            #for i in range(len(BERT_tokens)):
            #    print(BERT_tokens[i] + ' : ' + str(output_flags[i]))
            #input()
            
            #print('\n\n\n')           
            #for x in matched_pairs:
            #    print(x[0] + '\t: ' + x[1])
            #input()
            
            return BERT_tokens, BERT_ids, output_flags
    
    def get_BERT_embed(self, text, ALIGN=False, tgt_pos=['代名詞', '副詞', '助動詞', '助詞',
                                                   '動詞', '名詞', '形容詞', '感動詞',
                                                   '接尾辞', '接続詞', '接頭辞', '空白',
                                                   '補助記号', '記号', '連帯詞',
                                                   'フィラー']):
        
        if ALIGN:
            #word_list = word_align(text, tagger)
            word_list = self.normalize_mecab(text, tgt_pos=tgt_pos)
            text = ''.join(word_list)
        
        bert_tokens, token_ids = self.tokenize_BERT(text)
        
        if len(bert_tokens) > self.MAX_TOKEN_LEN:
            bert_tokens = bert_tokens[:self.MAX_TOKEN_LEN]
            token_ids = token_ids[:self.MAX_TOKEN_LEN]
        
        outputs = self.calc_BERT_embed(token_ids)
        return outputs
    
    def tokenize_BERT(self, sentence):
        #bert_tokens = bert_tokenizer.tokenize(" ".join(["[CLS]"] + output + ["[SEP]"]))
        input_ids = self.bert_tokenizer.encode(sentence, return_tensors='pt').to(self.device)
        bert_tokens = self.bert_tokenizer.convert_ids_to_tokens(input_ids[0])
        #print("BERT tokens: ")
        #for i in range(len(bert_tokens)):
        #    print(str(i) + bert_tokens[i])
        token_ids = self.bert_tokenizer.convert_tokens_to_ids(bert_tokens)
        #print("BERT token IDs: ", token_ids)
        return bert_tokens, token_ids
    
    def calc_BERT_embed(self, token_ids):
        """
        ベクトル取得
        """
        #print("\n *** to Vector ***")
        #print(token_ids)
        tokens_tensor = torch.tensor(token_ids).unsqueeze(0).to(self.device)
        # tokens_tensor = token_ids
        #print(np.shape(tokens_tensor))
        #outputs, _ = bert_model(tokens_tensor)
        outputs = self.bert_model(tokens_tensor).last_hidden_state
        #print(outputs)
        #print(type(outputs))
        #print(np.shape(outputs))
        #print(outputs[0], "\n (size: ", outputs[0].size(), ")")
        outputs = outputs.detach().to("cpu").numpy().copy()
        return outputs[0]
    
    def normalize_mecab(self, raw_sentence, tgt_pos=['代名詞', '副詞', '助動詞', '助詞',
                                               '動詞', '名詞', '形容詞', '感動詞',
                                               '接尾辞', '接続詞', '接頭辞', '空白',
                                               '補助記号', '記号', '連帯詞', 'フィラー']):
        sentence = self.tagger.parse(raw_sentence).split('\n')
        
        #eliminate 'EOS' and ''
        sentence.pop(-1)
        sentence.pop(-1)
        
        for i in range(len(sentence)):
            sentence[i] = sentence[i].split('\t')
            #print(sentence[i])
            sentence[i][1] = sentence[i][1].split(',')
    
        #sentence = [x[0] for x in sentence if (x[1][0] != 'フィラー') and (x[1][0] != '記号') and (x[1][0] != '感動詞') and (x[1][0] != '連体詞')]
        #sentence = [x[0] for x in sentence if (x[1][0] != '記号')]
        #sentence = [x[0] for x in sentence if ((x[1][0] == '名詞') or (x[1][0] == '動詞') or (x[1][0] == '形容詞') or (x[1][0] == '副詞'))]
        #sentence = [x[0] for x in sentence]
        
        output = []
        for x in sentence:
            if ((x[1][0] in tgt_pos) or (x[1][1] in tgt_pos)):
                output.append(x[0])
            
        return output
    
    def normalize_sudachi(self, text, 
                          tgt_pos = ['補助記号','空白','名詞','記号',
                                     '接頭辞','感動詞','副詞','接尾辞',
                                     '代名詞','形状詞','動詞','形容詞',
                                     '連体詞','接続詞',
                                     '助詞','助動詞']):
        
        tokenizer_obj = dictionary.Dictionary().create()
        mode = tokenizer_obj.SplitMode.C
            
        out = [x.normalized_form() for x in tokenizer_obj.tokenize(text, mode) if x.part_of_speech()[0] in tgt_pos]
            
        #x.surface()
        #x.reading_form()
        
        return out
    
    def tokenize_sudachi(self, text, 
                         tgt_pos = ['補助記号','空白','名詞','記号',
                                    '接頭辞','感動詞','副詞','接尾辞',
                                    '代名詞','形状詞','動詞','形容詞',
                                    '連体詞','接続詞',
                                    '助詞','助動詞'],
                         return_flags=False):
        tokenizer_obj = dictionary.Dictionary().create()
        mode = tokenizer_obj.SplitMode.C
        #out = tokenizer_obj.tokenize(text,mode)
        
        extracted_tokens = []
        flags = []
        raw_tokens = []
        for x in self.tokenizer_obj.tokenize(text, mode):
            if x.part_of_speech()[0] in tgt_pos:
                
                """
                # To capture fillers in Sudachi.
                # Another option is to use Ginza
                if x.part_of_speech()[0]:
                    if x.part_of_speech()[1] == 'フィラー':
                        print('フィラー！！！ > ', x.part_of_speech()[1])
                    else:
                        print('一般らしい・・・ > ', x.part_of_speech()[1])
                """
                
                x = str(x)
                extracted_tokens.append(x)
                flags.append(1)
                raw_tokens.append(x)
            else:
                x = str(x)
                flags.append(0)
                raw_tokens.append(x)
        #out = [x for x in tokenizer_obj.tokenize(text, mode) if x.part_of_speech()[0] in tgt_pos]
    
        #print(out)
        if return_flags:
            return extracted_tokens, flags, raw_tokens
        else:
            return extracted_tokens
    
    def check_seems_sorry(self, text):
        
        text = ''.join(text)
        flag = self._vocab_match(text, ['申し訳', '御免'])
        
        return [flag]
    
    """
    ####################
    --- Task : refuse
    ####################
    """
    def check_explicit_refuse(self, text):
    
        text = ''.join(text)
        flag = self._vocab_match(text, ['無理', '厳しい', 'できない'])
        #"できない"　どうやって検出
        
        return [flag]
    
    def _vocab_match(self, text, tgt_vocab_list):
        
        flag = 0
        
        words =[m for m in self.tokenizer_obj.tokenize(text, self.mode)]
        for word in words:
            
            dict_form = word.dictionary_form()
            norm_form = word.normalized_form()
            pos = word.part_of_speech()
            
            if DEBUG:
                print(dict_form)
                print(norm_form)
                print(pos)
            
            if norm_form in tgt_vocab_list:
                flag = 1
                break
            
        return flag
    
        
    
