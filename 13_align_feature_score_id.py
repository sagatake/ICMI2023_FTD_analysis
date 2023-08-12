#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:26:23 2023

@author: takeshi-s

TODO!
    prepared for ENGLISH feature headers

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
    
    ENGLISH = False
    # ENGLISH = True

    # score_dir   = Path('dataset/data_label/score/complete/')
    score_dir   = Path('dataset/data_label/score/incomplete/')
    
    feature_file = Path('features.csv')
    
    dream_flag_file = Path('dataset/data_label/dream_flag.csv')
    
    output_feature_file = Path('aligned_features_test.csv')

    output_score_file   = Path('aligned_scores_test.csv')

    
    assess_item_dict = {'BigFive':5, 'KiSS18':1, 'SPQ':10, 'SRS2':6}
    
    score_files = list(score_dir.iterdir())
    
    src_feature_list = _load_csv(feature_file)
    # print(src_feature_list)
    
    dream_flag_list = _load_csv(dream_flag_file)
    dream_flag_dict = {}
    for row in dream_flag_list[1:]:
        dream_flag_dict['{:09d}'.format(int(row[0]))] = row[1:]
        
    user_id_list = []

    if ENGLISH:
        feature_header_index = 1
    else:
        feature_header_index = 0
    
    feature_header = ['user_id', 'session_id']
    feature_header.extend(dream_flag_list[0][1:])
    feature_header.extend(src_feature_list[feature_header_index][2:])
    output_feature_list = [feature_header]
    for row in sorted(src_feature_list[2:]):
        
        try:
            _, user_id, _, session_id = row[0].split('_')
            tmp_row = [user_id, session_id]
            tmp_row.extend(dream_flag_dict[user_id])
            tmp_row.extend(row[2:])
            output_feature_list.append(tmp_row)
        except Exception as e:
            print(traceback.format_exc())
            pass
        
    pp.pprint(output_feature_list[:3])
    # print(np.shape(output_feature_list))
    
    # sys.exit()
        
    ###########################################################################
    
    output_score_list = []
    
    output_header_dict = {}
    output_score_dict = {}
    
    if ENGLISH:
        score_header_index = 1
    else:
        score_header_index = 0
    
    for score_file in score_files:
        
        print(score_file.stem)
        src_score_list = _load_csv(score_file)
        # pp.pprint(src_score_list)
        
        for row in src_score_list[2:]:
            
            user_id = '{:09d}'.format(int(row[0]))
            scores = row[1:]
            # input(scores)
            
            if user_id in output_score_dict.keys():
                output_score_dict[user_id][score_file.stem] = scores
            else:
                output_score_dict[user_id] = {}
                for assess_key in assess_item_dict.keys():
                    output_score_dict[user_id][assess_key] = ['N/A' for _ in range(assess_item_dict[assess_key])]
                output_score_dict[user_id][score_file.stem] = scores
                
            if not(score_file.stem in output_header_dict.keys()):
                header0 = [score_file.stem for _ in range(len(scores))]
                header1 = src_score_list[score_header_index][1:]
                output_header_dict[score_file.stem] = [header0, header1]
                
    # pp.pprint(output_score_dict)
    
    headers = [[], []]
    header_keys = output_header_dict.keys()
    for key in sorted(header_keys):        
        headers[0].extend(output_header_dict[key][0])
        headers[1].extend(output_header_dict[key][1])
    headers[0].insert(0, 'user_id')
    headers[1].insert(0, 'user_id')
    
    scores = []
    score_keys = output_score_dict.keys()
    for score_key in sorted(score_keys):
        row = []
        for header_key in sorted(header_keys):
            row.extend(output_score_dict[score_key][header_key])
        row.insert(0, score_key)
        # input(len(row))
        scores.append(row)
    

    output_score_list.extend(headers)
    output_score_list.extend(scores)
    
    # pp.pprint(output_score_dict)
    # pp.pprint(output_header_dict)
    # pp.pprint(headers)
    
    print(np.shape(headers))
    print(np.shape(scores))
    print(np.shape(output_score_list))
    
    if OUTPUT:
        _write_csv(output_feature_file, output_feature_list)
        _write_csv(output_score_file, output_score_list)
        
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

