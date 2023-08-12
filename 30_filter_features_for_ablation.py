#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:40:23 2023

@author: takeshi-s
"""
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
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

def main():
    "Main function"
    
    OUTPUT = False
    # OUTPUT = True
    
    src_file = Path('aligned_features.csv')
    # src_file = Path('aligned_features_separated180.csv')
    
    src_data = _load_csv(src_file)
    
    
    for i in range(len(src_data[0])):
        print(i, src_data[0][i])
    # pp.pprint(src_data1[0])

    # CDI = pos_dict['連体詞'] + pos_dict['助詞'] - pos_dict['代名詞'] - (pos_dict['助動詞'] + pos_dict['接尾辞']) 
    #     - pos_dict['接続詞'] - negation_weight * num_negation - pos_dict['副詞']
    # tgt_col_index_list = [0, 1, 2, 3, 4,
    #                       18, 20, 14, 21, 13,
    #                       19, 5, 12]
        
    # Full
    tgt_file = Path('aligned_features_full.csv')
    tgt_col_index_list = [0, 1, 2, 3, 4,5, 
                            # 6,
                            7,8,9,          #BERT(sent-diff, sent-cosine, sent-cont)
                            # 10,11,
                            12,             #abst: CDI-J
                            13,             #func
                            # 14,15         #補助記号, 空白
                            16,             #cont
                            # 17,           #記号
                            18, 19,         #func
                            20,             #cont
                            21,22,          #func
                            23, 24, 25,     #cont
                            26, 27, 28, 29, #func
                            30,             #abst: ratio_cont
                            31,32 ,         #temporal: ratio_punct, WPM
                          ]

    # # w/o embedding(BERT)
    # tgt_file = Path('aligned_features_wo-BERT.csv')
    # tgt_col_index_list = [0, 1, 2, 3, 4,5, 
    #                         # 6,
    #                         # 7,8,9,          #BERT(sent-diff, sent-cosine, sent-cont)
    #                         # 10,11,
    #                         12,             #abst: CDI-J
    #                         13,             #func
    #                         # 14,15         #補助記号, 空白
    #                         16,             #cont
    #                         # 17,           #記号
    #                         18, 19,         #func
    #                         20,             #cont
    #                         21,22,          #func
    #                         23, 24, 25,     #cont
    #                         26, 27, 28, 29, #func
    #                         30,             #abst: ratio_cont
    #                         31,32 ,         #temporal: ratio_punct, WPM
    #                       ]

    # # w/o content word
    # tgt_file = Path('aligned_features_wo-cont.csv')
    # tgt_col_index_list = [0, 1, 2, 3, 4,5, 
    #                         # 6,
    #                         7,8,9,          #BERT(sent-diff, sent-cosine, sent-cont)
    #                         # 10,11,
    #                         12,             #abst: CDI-J
    #                         13,             #func
    #                         # 14,15         #補助記号, 空白
    #                         # 16,             #cont
    #                         # 17,           #記号
    #                         18, 19,         #func
    #                         # 20,             #cont
    #                         21,22,          #func
    #                         # 23, 24, 25,     #cont
    #                         26, 27, 28, 29, #func
    #                         30,             #abst: ratio_cont
    #                         31,32 ,         #temporal: ratio_punct, WPM
    #                       ]

    # # w/o function word
    # tgt_file = Path('aligned_features_wo-func.csv')
    # tgt_col_index_list = [0, 1, 2, 3, 4,5, 
    #                         # 6,
    #                         7,8,9,          #BERT(sent-diff, sent-cosine, sent-cont)
    #                         # 10,11,
    #                         12,             #abst: CDI-J
    #                         # 13,             #func
    #                         # 14,15         #補助記号, 空白
    #                         16,             #cont
    #                         # 17,           #記号
    #                         # 18, 19,         #func
    #                         20,             #cont
    #                         # 21,22,          #func
    #                         23, 24, 25,     #cont
    #                         # 26, 27, 28, 29, #func
    #                         30,             #abst: ratio_cont
    #                         31,32 ,         #temporal: ratio_punct, WPM
    #                       ]

    # # w/o abstract
    # tgt_file = Path('aligned_features_wo-abst.csv')
    # tgt_col_index_list = [0, 1, 2, 3, 4,5, 
    #                         # 6,
    #                         7,8,9,          #BERT(sent-diff, sent-cosine, sent-cont)
    #                         # 10,11,
    #                         # 12,             #abst: CDI-J
    #                         13,             #func
    #                         # 14,15         #補助記号, 空白
    #                         16,             #cont
    #                         # 17,           #記号
    #                         18, 19,         #func
    #                         20,             #cont
    #                         21,22,          #func
    #                         23, 24, 25,     #cont
    #                         26, 27, 28, 29, #func
    #                         # 30,             #abst: ratio_cont
    #                         31,32 ,         #temporal: ratio_punct, WPM
    #                       ]

    # # w/o temporal
    # tgt_file = Path('aligned_features_wo-temporal.csv')
    # tgt_col_index_list = [0, 1, 2, 3, 4,5, 
    #                         # 6,
    #                         7,8,9,          #BERT(sent-diff, sent-cosine, sent-cont)
    #                         # 10,11,
    #                         12,             #abst: CDI-J
    #                         13,             #func
    #                         # 14,15         #補助記号, 空白
    #                         16,             #cont
    #                         # 17,           #記号
    #                         18, 19,         #func
    #                         20,             #cont
    #                         21,22,          #func
    #                         23, 24, 25,     #cont
    #                         26, 27, 28, 29, #func
    #                         30,             #abst: ratio_cont
    #                         # 31,32 ,         #temporal: ratio_punct, WPM
    #                       ]

    
    tgt_list = []
    tgt_name_list = []
    
    for tgt_index in tgt_col_index_list:
        tgt_name_list.append(src_data[0][tgt_index])
    
    for i in range(1, len(src_data)):
        row = []
        for tgt_index in tgt_col_index_list:
            row.append(src_data[i][tgt_index])
        tgt_list.append(row)
    
    tgt_list.insert(0, tgt_name_list)
    
    pp.pprint(tgt_list[:2])
    
    if OUTPUT:
        _write_csv(tgt_file, tgt_list)

def _load_csv(file_name, encoding = 'utf-8'):
    with open(file_name, encoding = encoding) as f:
        reader = csv.reader(f)
        data = [x for x in reader]
    return data

def _write_csv(file_name, data, encoding = 'utf-8'):
    with open(file_name, 'w', encoding = encoding) as f:
        writer = csv.writer(f)
        writer.writerows(data)

def _change_csv_encoding(src_file_name, tgt_file_name, src_encoding, tgt_encoding):
    with open(src_file_name, encoding = src_encoding) as f:
        reader = csv.reader(f)
        data = [x for x in reader]
    with open(tgt_file_name, 'w', encoding = tgt_encoding) as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
if __name__ == '__main__':
    main()

