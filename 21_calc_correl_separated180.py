#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:26:23 2023

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

from scipy.stats import spearmanr

def main():
    "Main function"
    
    feature_file    = Path('aligned_features_separated180_cp932.csv')
    
    score_file      = Path('aligned_scores_separated180_cp932.csv')
    
    output_file = Path('result_correl_separated180_cp932.csv')
    
    output_full_file = Path('aligned_features_scores_separated180_cp932.csv')
    
    features    = _load_csv(feature_file, 'cp932')
    scores      = _load_csv(score_file, 'cp932')
    
    output_full_list = []
    row = []
    row.extend(features[0][:6])
    for i in range(1, len(scores[0])):
        row.append('{}_{}'.format(scores[0][i], scores[1][i]))
    row.extend(features[0][6:])
    output_full_list.append(row)
    for i in range(1, len(features)):
        row = []
        row.extend(features[i][:6])
        for j in range(2, len(scores)):
            if scores[j][0] == features[i][0]:
                row.extend(scores[j][1:])
        row.extend(features[i][6:])
        output_full_list.append(row)
    # pp.pprint(output_full_list[:3])
    # sys.exit()
    _write_csv(output_full_file, output_full_list, 'cp932')
    
    features    = np.asarray(features)
    scores      = np.asarray(scores)

    
    # pp.pprint(features[:3])
    # sys.exit()
    
    # feature_user_ids        = features[:, 0]
    # feature_sess_ids        = features[:, 1]
    # feature_dream_flags     = features[:, 2:5]
    # feature_vals            = features[:, 5:]
    
    # score_user_ids  = scores[:, 0]
    # score_BigFive   = scores[:, 1:6]
    # score_KiSS18    = scores[:, 6]
    # score_SPQ       = scores[:, 7:17]
    # score_SRS2      = scores[:, 17:]
    
    # print(np.shape(score_user_ids))
    # print(np.shape(score_BigFive))
    # print(np.shape(score_KiSS18))
    # print(np.shape(score_SPQ))
    # print(np.shape(score_SRS2))
    
    # Session 1, 4, 7: dream or favorite thing
    feature_dream_favorite = feature_filter(features, [1], [1,4,7])
    
    # Session 1, 4, 7: dream
    feature_dream = feature_filter(feature_dream_favorite, [3, 4, 5], [1])
    tmp = feature_dream

    # Session 1, 4, 7: favorite thing
    feature_favorite = feature_filter(feature_dream_favorite, [3, 4, 5], [0])
    tmp = feature_favorite

    # Session 2, 5, 8: negative memory
    feature_negative = feature_filter(features, [1], [2, 5, 8])
    tmp = feature_negative
    
    # Session 3, 6, 9: biggest mistake
    feature_mistake = feature_filter(features, [1], [3, 6, 9])
    tmp = feature_mistake

    # feature_30sec = feature_filter(features, [2], [1])
    # feature_60sec = feature_filter(features, [2], [2])
    # feature_180sec = feature_filter(features, [2], [3])

    # feature_30sec_neg = feature_filter(feature_negative, [2], [1])
    # feature_60sec_neg = feature_filter(feature_negative, [2], [2])
    # feature_180sec_neg = feature_filter(feature_negative, [2], [3])

    # feature_30sec = feature_filter(features, [2], [1])
    # feature_60sec = feature_filter(features, [2], [2])
    feature_180sec = feature_filter(features, [2], [1,2,3])

    # feature_30sec_neg = feature_filter(feature_negative, [2], [1])
    # feature_60sec_neg = feature_filter(feature_negative, [2], [2])
    feature_180sec_neg = feature_filter(feature_negative, [2], [1,2,3])

    
    # pp.pprint(tmp[:, :6])
    # print(len(tmp)-1)
    # sys.exit()
    
    # tgt_feature_indexes = list(range(6, 12))
    tgt_feature_indexes = list(range(6, len(features[0])))
    
    print('features')
    print(np.shape(feature_dream_favorite))
    print(np.shape(feature_dream))
    print(np.shape(feature_favorite))
    print(np.shape(feature_negative))
    print(np.shape(feature_mistake))
    # print(np.shape(feature_30sec))
    # print(np.shape(feature_60sec))
    print(np.shape(feature_180sec))
    print(np.shape(feature_180sec_neg))
    
    # pp.pprint(feature_180sec[:3])
    
    tgt_score = None
    tgt_subscore = None
    
    # tgt_score = 'SPQ'
    # tgt_subscore = '奇異な話し方'
    # tgt_subscore = 'SPQ'

    # tgt_score = 'SRS2'
    
    print('tgt_score: ', tgt_score)
    print('tgt_subscore: ', tgt_subscore)
    
    output_list = [['group', 'score', 'subscore', 'feature', 'spearman_r', 'p-val', 'significant05', 'significant10']]

    # tag = 'dream'        
    # # print('################')
    # # print('### {} '.format(tag))
    # # print('################')
    # # feature_main = pick_columns(feature_dream, tgt_feature_indexes)    
    # # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # # for i in range(len(result_list)):
    # #     result_list[i].insert(0, tag)
    # # output_list.extend(result_list)
    # output_list = calc_correl_ouptut(tag, output_list, feature_dream, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)
    
    # tag = 'favorite'        
    # # print('################')
    # # print('### {} '.format(tag))
    # # print('################')
    # # feature_main = pick_columns(feature_favorite, tgt_feature_indexes)    
    # # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # # for i in range(result_list):
    # #     result_list[i].insert(0, tag)
    # # output_list.extend(result_list)
    # output_list = calc_correl_ouptut(tag, output_list, feature_favorite, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)

    # tag = 'negative'        
    # # print('################')
    # # print('### {} '.format(tag))
    # # print('################')
    # # feature_main = pick_columns(feature_negative, tgt_feature_indexes)    
    # # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # # for i in range(result_list):
    # #     result_list[i].insert(0, tag)
    # # output_list.extend(result_list)
    # output_list = calc_correl_ouptut(tag, output_list, feature_negative, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)

    # tag = 'mistake'        
    # # print('################')
    # # print('### {} '.format(tag))
    # # print('################')
    # # feature_main = pick_columns(feature_mistake, tgt_feature_indexes)    
    # # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # # for i in range(result_list):
    # #     result_list[i].insert(0, tag)
    # # output_list.extend(result_list)
    # output_list = calc_correl_ouptut(tag, output_list, feature_mistake, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)



    # tag = '30sec'        
    # # print('################')
    # # print('### {} '.format(tag))
    # # print('################')
    # # feature_main = pick_columns(feature_30_sec, tgt_feature_indexes)    
    # # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # # for i in range(result_list):
    # #     result_list[i].insert(0, tag)
    # # output_list.extend(result_list)
    # output_list = calc_correl_ouptut(tag, output_list, feature_30sec, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)

    # tag = '60sec'        
    # # print('################')
    # # print('### {} '.format(tag))
    # # print('################')
    # # feature_main = pick_columns(feature_60_sec, tgt_feature_indexes)    
    # # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # # for i in range(result_list):
    # #     result_list[i].insert(0, tag)
    # # output_list.extend(result_list)
    # output_list = calc_correl_ouptut(tag, output_list, feature_60sec, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)

    tag = '180sec'        
    # print('################')
    # print('### {} '.format(tag))
    # print('################')
    # feature_main = pick_columns(feature_180_sec, tgt_feature_indexes)    
    # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # for i in range(result_list):
    #     result_list[i].insert(0, tag)
    # output_list.extend(result_list)
    output_list = calc_correl_ouptut(tag, output_list, feature_180sec, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)



    # tag = '30sec_neg'        
    # # print('################')
    # # print('### {} '.format(tag))
    # # print('################')
    # # feature_main = pick_columns(feature_30_sec_neg, tgt_feature_indexes)    
    # # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # # for i in range(result_list):
    # #     result_list[i].insert(0, tag)
    # # output_list.extend(result_list)
    # output_list = calc_correl_ouptut(tag, output_list, feature_30sec_neg, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)

    # tag = '60sec_neg'        
    # # print('################')
    # # print('### {} '.format(tag))
    # # print('################')
    # # feature_main = pick_columns(feature_60_sec_neg, tgt_feature_indexes)    
    # # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # # for i in range(result_list):
    # #     result_list[i].insert(0, tag)
    # # output_list.extend(result_list)
    # output_list = calc_correl_ouptut(tag, output_list, feature_60sec_neg, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)

    tag = '180sec_neg'        
    # print('################')
    # print('### {} '.format(tag))
    # print('################')
    # feature_main = pick_columns(feature_180_sec_neg, tgt_feature_indexes)    
    # result_list = calc_correl(feature_main, scores, tgt_score, tgt_subscore)
    # for i in range(result_list):
    #     result_list[i].insert(0, tag)
    # output_list.extend(result_list)
    output_list = calc_correl_ouptut(tag, output_list, feature_180sec_neg, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list)

    _write_csv(output_file, output_list, 'cp932')

def calc_correl_ouptut(major_tag, output_list, features, tgt_feature_indexes, scores, tgt_score, tgt_subscore, output_full_list):
    
    # tag = 'dream'        
    print('################')
    print('### {} '.format(major_tag))
    print('################')
    for phase_index in [1,2,3]:
        
        tag = '{}_{}'.format(major_tag, phase_index)
        phase_features = phase_filter(features, phase_index)
        print('phase', np.shape(phase_features))
        feature_main = pick_columns(phase_features, tgt_feature_indexes)
        print('main', np.shape(feature_main))
        
        result_list = calc_correl(tag, output_full_list, feature_main, scores, tgt_score, tgt_subscore)
        for i in range(len(result_list)):
            result_list[i].insert(0, tag)
        output_list.extend(result_list)
    
    return output_list

def phase_filter(src, tgt_phase):
    
    tgt = [src[0]]
    for row in src[1:]:
        # input(row)
        if row[2] == str(tgt_phase):
            tgt.append(row)
            
    # pp.pprint(tgt[:3])
    # print(tgt_phase)
    # sys.exit()
    
    tgt = np.asarray(tgt)
    
    return tgt

def pick_columns(src, tgt_indexes):
    
    # tgt = []
    user_ids = src[:, 0].reshape(-1, 1)
    tgt = user_ids
    # tgt.append(user_ids)
    
    # tgt = np.asarray(tgt)
    
    
    for tgt_index in tgt_indexes:
        tgt = np.hstack([tgt, src[:, tgt_index].reshape(-1, 1)])

    # print(np.shape(tgt))
    # pp.pprint(tgt[:2])
    # sys.exit()
    
    return tgt
    
def calc_correl(tag, output_full_list, features, scores, tgt_score = None, tgt_subscore = None, p_thread = 0.1):
    
    features = features.tolist()
    features.insert(0, features[0])
    
    scores = scores.tolist()
    
    aligned_scores = [scores[0], scores[1]]
    for feature_row in features[2:]:
        for score_row in scores[2:]:
            if feature_row[0] == score_row[0]:
                aligned_scores.append(score_row)
    
    # print('features, scores')
    # print(np.shape(features))
    # print(np.shape(aligned_scores))
    
    tgt_features    = [features[0], features[1]]
    tgt_scores      = [aligned_scores[0], aligned_scores[1]]
    for i in range(2, len(features)):
        if 'N/A' in aligned_scores[i]:
            pass
        else:
            tgt_features.append(features[i])
            tgt_scores.append(aligned_scores[i])
        # input(aligned_scores[i])
            
    tgt_features = np.asarray(tgt_features)
    tgt_scores = np.asarray(tgt_scores)
    
    print('tgt_features, tgt_scores')
    print(np.shape(tgt_features))
    print(np.shape(tgt_scores))
    
    output_list = []
    
    for i in range(1, len(tgt_scores[0])):
        for j in range(1, len(tgt_features[0])):
                        
            stat, p_val = spearmanr(tgt_features[2:, j], tgt_scores[2:, i])
            
            significant_05 = 'Yes' if p_val < 0.05 else ''
            significant_10 = 'Yes' if p_val < 0.10 else ''
            
            row = [tgt_scores[0][i], tgt_scores[1][i], tgt_features[0][j], stat, p_val, significant_05, significant_10]
            output_list.append(row)

            if (p_val < p_thread) and (tgt_score == None) and (tgt_subscore == None):
                print('##### ', end='')
                print(tag, '{:>6.3f}, {:03.3f}, {:>10}_{:}, {:}'.format(stat, p_val, tgt_scores[0][i], tgt_scores[1][i], tgt_features[0][j]))
            elif (p_val < p_thread) and (tgt_scores[0][i] == tgt_score):
                print('##### ', end='')
                print(tag, '{:>6.3f}, {:03.3f}, {:>10}_{:}, {:}'.format(stat, p_val, tgt_scores[0][i], tgt_scores[1][i], tgt_features[0][j]))
            elif (p_val < p_thread) and (tgt_score == None) and (tgt_scores[1][i] == tgt_subscore):
                print('##### ', end='')
                print(tag, '{:>6.3f}, {:03.3f}, {:>10}_{:}, {:}'.format(stat, p_val, tgt_scores[0][i], tgt_scores[1][i], tgt_features[0][j]))
            
            # print(tgt_features[2:, j])
            # print(tgt_scores[2:, i])
            # print(tag, '{:>6.3f}, {:03.3f}, {:>10}_{:}, {:}'.format(stat, p_val, tgt_scores[0][i], tgt_scores[1][i], tgt_features[0][j]))
            # input()
    
    # return stat, p_val
    return output_list
    
def feature_filter(src, tgt_indexes, tgt_keys):
    
    if type(tgt_keys[0]) != type('a'):
        tgt_keys = [str(x) for x in tgt_keys]
    
    tgt_list = [src[0].tolist()]
    for row in src[1:]:
        for tgt_index in tgt_indexes:
            
            if row[tgt_index] in tgt_keys:
                
                # Dream flag filtering
                if tgt_index == 3:
                    if row[1] == '1':
                        tgt_list.append(row.tolist())
                elif tgt_index == 4:
                    if row[1] == '4':
                        tgt_list.append(row.tolist())
                elif tgt_index == 5:
                    if row[1] == '7':
                        tgt_list.append(row.tolist())   
                
                # others
                else:
                    tgt_list.append(row.tolist())
    
    tgt_array = np.asarray(tgt_list)
    
    return tgt_array
                
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

