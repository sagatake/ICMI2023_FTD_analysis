#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:26:23 2023

@author: takeshi-s

#########################
##### Not completed
#########################
"""
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pprint as pp
import pandas as pd
import numpy as np
import traceback
import shutil
import random
import copy
import math
import time
import csv
import sys
import os

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import GridSearchCV

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

TQDM = True

import multiprocessing
# NUM_WORKER = multiprocessing.cpu_count()
NUM_WORKER = 1

def main():
    "Main function"
    
    feature_file    = Path('aligned_features.csv')
    # feature_file    = Path('aligned_features_wo-BERT.csv')
    # feature_file    = Path('aligned_features_wo-cont.csv')
    # feature_file    = Path('aligned_features_wo-func.csv')
    # feature_file    = Path('aligned_features_wo-abst.csv')
    # feature_file    = Path('aligned_features_wo-temporal.csv')
    #
    # feature_file    = Path('aligned_features_w-func-abst.csv')
    # feature_file    = Path('aligned_features_w-func-abst-cont.csv')
    # feature_file    = Path('aligned_features_w-func-abst-temporal.csv')
    # feature_file    = Path('aligned_features_w-BERT-func-cont.csv')

    
    score_file      = Path('aligned_scores.csv')
    
    # output_file     = Path('result_estimation_cp932.csv')
    
    output_coef_file = Path('result_coef_cp932.csv')
    
    features    = _load_csv(feature_file)
    scores      = _load_csv(score_file)
    
    print(np.shape(features))
    
    # tgt_score_indexes = [1, 2, 3, 4, 5]                           #BigFive
    # tgt_score_indexes = [6]                                       #KiSS18
    # tgt_score_indexes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]     #SPQ
    tgt_score_indexes = [7]                                       #SPQ_SPQ
    # tgt_score_indexes = [16]                                      #SPQ_OddSpeech
    # tgt_score_indexes = [17]                                      #SRS
    
    scores = pick_columns(scores, tgt_score_indexes)
    scores = filter_NA(scores)
    
    print(np.shape(scores))
    
    # Session 1=dream/fav(30), 2=negative(30), 3=mistake(30)
    tgt_sess_indexes = None
    # tgt_sess_indexes = [7, 8, 9] #30
    # tgt_sess_indexes = [7, 8, 9] #60
    # tgt_sess_indexes = [7, 8, 9] #180
    # tgt_sess_indexes = [4, 5, 6, 7, 8, 9] #60&180
    tgt_sess_indexes = [2, 5, 8] #negative
    # tgt_sess_indexes = [3, 6, 9] #mistake
    # tgt_sess_indexes = [1, 4, 7] #dream/favorite
    # tgt_sess_indexes = [1, 2, 4, 5, 7, 8] #dream/negative

    # tgt_sess_indexes = [2] #negative, 30
    # tgt_sess_indexes = [5] #negative, 60
    # tgt_sess_indexes = [8] #negative, 180
    # tgt_sess_indexes = [5, 8] #negative, 60&180
    # tgt_sess_indexes = [5, 6, 8, 9] #negative&mistake, 60&18
    
    DREAM = None
    # DREAM = False
    # DREAM = True
    
    WITH_TASK_DURATION = False
    # WITH_TASK_DURATION = True
    
    WITH_FOUR_PAIR = False
    # WITH_FOUR_PAIR = True
    
    SELECT_N_FEATURE = None
    SELECT_N_FEATURE = 10

    
    # model = LinearRegression()
    # model = Lasso()
    # model = RandomForestRegressor(max_features = 'sqrt', n_jobs = NUM_WORKER)
    # model = RandomForestRegressor(n_jobs = NUM_WORKER)
    # model = PLSRegression(scale = False, n_components=1)
    # model = PLSRegression(scale = False, n_components=7)
    # model = SVR()
    
    model = PLSRegression(scale = False)
    parameters = {'n_components':[1,2,3,4,5,6,7,8,9,10]}
    model = GridSearchCV(model, parameters, cv=5)
    best_params_list = []

    aligned_features = []
    aligned_scores = []
    
    if WITH_TASK_DURATION:
        features = add_task_duration(features)
        
    if WITH_FOUR_PAIR:
        features = make_four_pair(features, tgt_sess_indexes)
        aligned_features.append(features[0])
        aligned_scores.append(scores[0])
        for i in range(1, len(features)):
            for j in range(2, len(scores)):
                if features[i][0] == scores[j][0]:
                    aligned_features.append(features[i])
                    aligned_scores.append(scores[j])
    
    else:
        for i in range(1, len(features)):
            
            user_id = features[i][0]
            sess_id = features[i][1]
            dream_flag1 = features[i][2]
            dream_flag4 = features[i][3]
            dream_flag7 = features[i][4]
            
            if (tgt_sess_indexes != None):
                if not(int(sess_id) in tgt_sess_indexes):
                    continue
                if DREAM != None:
                    if (sess_id == '1') and (DREAM != bool(int(dream_flag1))):
                        continue
                    elif (sess_id == '4') and (DREAM != bool(int(dream_flag4))):
                        continue
                    elif (sess_id == '7') and (DREAM != bool(int(dream_flag7))):
                        continue

            row_features = ['{}_{}'.format(user_id, sess_id)]
            row_features.extend(features[i][5:])
            
            for j in range(len(scores)):
    
                row_scores = scores[j].copy()
                
                if user_id == row_scores[0]:
                    
                    row_scores[0] = row_features[0] 
                    
                    aligned_features.append(row_features)
                    aligned_scores.append(row_scores)
                    # print(user_id, row_scores[0])
                else:
                    pass
            
            # input(user_id)
    
        features = pick_columns(features, list(range(5, len(features[0]))))
        
        feature_names = features[0].copy()
        score_names = scores[0].copy()
        
        aligned_features.insert(0, feature_names)
        aligned_scores.insert(0, score_names)
    
    print(np.shape(aligned_features))
    print(np.shape(aligned_scores))

    ###########################################################################
    
    # sys.exit()
    
    print('##### Data inspection #####')
    # pp.pprint(aligned_features[:2])
    for i in range(2):
        for j in range(len(aligned_features[0])):
            print(j, aligned_features[i][j])
    pp.pprint(aligned_scores[:2])
    
    true_pred_list = [['true', 'pred']]
    
    index_list = list(range(1, len(aligned_features)))
    random.shuffle(index_list)

    coef_list = []

    for i in tqdm(index_list, disable=not(TQDM)):
        
        if WITH_FOUR_PAIR:
            user_id             = aligned_features[i][0]
        else:
            data_id             = aligned_features[i][0]
            user_id, sess_id    = data_id.split('_')
        
        test_X = np.asarray([aligned_features[i][1:]], dtype = np.float32)
        test_y = np.asarray([aligned_scores[i][1]], dtype = np.float32)
        
        # Leave one participant out
        train_X, train_y = leave_one_participant_out(aligned_features, aligned_scores, user_id)
        
        
        
        # Leave one subject out
        # train_X, train_y = aligned_features.copy(), aligned_scores.copy()
        # train_X.pop(i)
        # train_y.pop(i)
        # train_X = [X[1:] for X in train_X[1:]]
        # train_y = [y[1:] for y in train_y[1:]]
        
        normalizer = StandardScaler()
        train_X = normalizer.fit_transform(train_X)
        test_X = normalizer.transform(test_X)
        
        # pt = PowerTransformer()
        # train_y = pt.fit_transform(np.reshape(train_y, (-1, 1)))
        
        # print(np.shape(train_X))
        # print(np.shape(train_y))
        # print(np.shape(test_X))
        # print(np.shape(test_y))
                
        tmp_model = copy.deepcopy(model)
        
        tmp_model.fit(train_X, train_y)
        
        pred_y = tmp_model.predict(test_X)
        
        
        # pred_y = pt.inverse_transform(pred_y)
        
        # print(data_id, test_y, pred_y)
        # input()
        
        if tmp_model.__class__ == RandomForestRegressor:
            # coef_list.append(tmp_model.feature_importances_)
            tmp_coef_list = tmp_model.feature_importances_
            true_pred_list.append([test_y[0], pred_y[0]])
        elif (tmp_model.__class__ == LinearRegression) or (tmp_model.__class__ == Lasso):
            # coef_list.append(tmp_model.coef_)
            tmp_coef_list = tmp_model.coef_
            true_pred_list.append([test_y[0], pred_y[0]])
        elif (tmp_model.__class__ == SVR):
            true_pred_list.append([test_y[0], pred_y[0]])
        elif (tmp_model.__class__ == GridSearchCV) and (tmp_model.best_estimator_.__class__ == PLSRegression):
            # coef_list.append(tmp_model.best_estimator_.coef_)
            tmp_coef_list = tmp_model.best_estimator_.coef_
            best_params_list.append(tmp_model.best_params_)
            true_pred_list.append([test_y[0], pred_y[0][0]])            
        else:
            # coef_list.append(tmp_model.coef_)
            tmp_coef_list = tmp_model.coef_
            true_pred_list.append([test_y[0], pred_y[0][0]])
            
        if SELECT_N_FEATURE != None:
            # tmp_coef_list = sorted(tmp_coef_list, key = lambda x:abs(x), reverse = True)
            # print(tmp_coef_list)
            if (model.__class__ == RandomForestRegressor) or (model.__class__ == LinearRegression) or (model.__class__ == Lasso):
                tmp = [[tmp_coef_list[i], aligned_features[0][i+1]] for i in range(len(tmp_coef_list))]
            else:
                tmp = [[tmp_coef_list[i][0], aligned_features[0][i+1]] for i in range(len(tmp_coef_list))]
            tmp_coef_list = sorted(tmp, key = lambda x:abs(x[0]), reverse=True)
            tmp_coef_list = tmp_coef_list[:SELECT_N_FEATURE]
            # pp.pprint(tmp_coef_list)
            
            # sys.exit()

        coef_list.append(tmp_coef_list)

        
        
    true_pred_array = np.asarray(true_pred_list[1:])
    
    if (model.__class__ != SVR):
        if (SELECT_N_FEATURE == None):
        
            ave_coef = np.average(coef_list, axis=0)
            # print(ave_coef)
            if (model.__class__ == RandomForestRegressor) or (model.__class__ == LinearRegression) or (model.__class__ == Lasso):
                tmp = [[ave_coef[i], aligned_features[0][i+1]] for i in range(len(ave_coef))]
            else:
                tmp = [[ave_coef[i][0], aligned_features[0][i+1]] for i in range(len(ave_coef))]
            tmp = sorted(tmp, key = lambda x:abs(x[0]), reverse=True)
            for i in range(len(tmp)):
                # print(tmp[i][0])
                # print(tmp[i][1])
                print('{:7.3f}, {}'.format(tmp[i][0], tmp[i][1]))
            
            _write_csv(output_coef_file, tmp, 'cp932')
            
        else:
            
            freq_dict = {}
            for i in range(len(coef_list)):
                for j in range(len(coef_list[0])):
                    key = coef_list[i][j][1]
                    if key in freq_dict.keys():
                        freq_dict[key] += 1
                    else:
                        freq_dict[key] = 1
            tmp = [[key, freq_dict[key]] for key in freq_dict.keys()]
            tmp = sorted(tmp, key = lambda x:x[1], reverse = True)
            print('Top features (N:{})'.format(SELECT_N_FEATURE))
            pp.pprint(tmp[:10])
    
    RMSE            = mean_squared_error(true_pred_array[:, 0], true_pred_array[:, 1], squared = False)
    R2              = r2_score(true_pred_array[:, 0], true_pred_array[:, 1])
    correl, p_val   = spearmanr(true_pred_array[:, 0], true_pred_array[:, 1])
    
    print('RMSE: {}, R2: {}, Correlation: {} ({:.3f})'.format(RMSE, R2, correl, p_val))
    
    best_params_dict = {}
    for key in best_params_list[0].keys():
        best_params_dict[key] = []
    for param_pair in best_params_list:
        for key in param_pair.keys():
            best_params_dict[key].append(param_pair[key])
    for key in best_params_dict.keys():
        best_params_dict[key] = np.average(best_params_dict[key])
    pp.pprint(best_params_dict)

    _write_csv('test_cp932.csv', true_pred_list, 'cp932')

    # _write_csv(output_file, output_list, 'cp932')

def make_four_pair(src, tgt_sess_indexes):
    
    task_ids = ['negative', 'mistake', 'dream', 'favorite']
    names   = src[0].copy()
    id_dict = {}
    max_len = 0
    
    for i in range(1, len(src)):
        
        # print(src[1])
        
        user_id     = src[i][0]
        sess_id     = int(src[i][1])
        dream_flag1 = src[i][2]
        dream_flag4 = src[i][3]
        dream_flag7 = src[i][4]
        features    = src[i][5:]

        if not(user_id in id_dict.keys()):
            id_dict[user_id] = {'negative': [], 'mistake': [], 'dream': [], 'favorite': []}
        
        #dream/favorite
        if sess_id in [1, 4, 7]:

            if (sess_id == 1):
                if bool(int(dream_flag1)):
                    id_dict[user_id]['dream'] = features
                else:
                    id_dict[user_id]['favorite'] = features

            elif (sess_id == 4):
                if bool(int(dream_flag4)):
                    id_dict[user_id]['dream'] = features
                else:
                    id_dict[user_id]['favorite'] = features

            elif (sess_id == 7):
                if bool(int(dream_flag7)):
                    id_dict[user_id]['dream'] = features
                else:
                    id_dict[user_id]['favorite'] = features
                
        #negative
        elif sess_id in [2, 5, 8]:
            id_dict[user_id]['negative'] = features
        
        #mistake
        elif sess_id in [3, 6, 9]:
            id_dict[user_id]['mistake'] = features
        
        if max_len < len(features):
            max_len = len(features)
    
    tgt = []
    row = []
    row.append(names[0])
    for task_id in task_ids:
        for name in names[5:]:
            row.append('{}_{}'.format(task_id, name))
    tgt.append(row)
    
    for user_id in id_dict.keys():
        for task_id in task_ids:
            if len(id_dict[user_id][task_id]) == 0:
                id_dict[user_id][task_id] = [0 for _ in range(max_len)]
                # print(len(id_dict[user_id][task_id]))
        row = [user_id]
        for task_id in task_ids:
            row.extend(id_dict[user_id][task_id])
        # print(len(row))
        tgt.append(row)
    
    # pp.pprint(tgt[:2])
    # print(np.shape(tgt))
    # sys.exit()
    
    return tgt
        
            
def add_task_duration(src):
    
    # pp.pprint(src[:2])
    
    tgt = []
    
    row = src[0].copy()
    row.insert(5, 'task')
    row.insert(6, 'duration')
    tgt.append(row)
    
    for i in range(1, len(src[1:])):
        
        row = [src[i][0]]
        if (src[i][1] == '1') and (src[i][2] == '0'):
            row.extend([1, 1])
        elif (src[i][1] == '1') and (src[i][2] == '1'):
            row.extend([2, 1])
        elif (src[i][1] == '2'):
            row.extend([3, 1])
        elif (src[i][1] == '3'):
            row.extend([4, 1])
        elif (src[i][1] == '4') and (src[i][3] == '0'):
            row.extend([1, 2])
        elif (src[i][1] == '4') and (src[i][3] == '1'):
            row.extend([2, 2])
        elif (src[i][1] == '5'):
            row.extend([3, 2])
        elif (src[i][1] == '6'):
            row.extend([4, 2])
        elif (src[i][1] == '7') and (src[i][4] == '0'):
            row.extend([1, 3])
        elif (src[i][1] == '7') and (src[i][4] == '1'):
            row.extend([2, 3])
        elif (src[i][1] == '8'):
            row.extend([3, 3])
        elif (src[i][1] == '9'):
            row.extend([4, 3])
        row.extend(src[i][1:])
        tgt.append(row)
    
    # pp.pprint(tgt[:2])
    # sys.exit()
    
    return tgt
    
def leave_one_participant_out(src_features, src_scores, user_id):
    
    # tgt_features = [src_features[0]]
    # tgt_scores = [src_scores[0]]

    tgt_features = []
    tgt_scores = []
    
    for i in range(1, len(src_features)):
        if user_id in src_features[i][0]:
            pass
        else:
            # print(np.shape(src_features[i][1:]))
            tgt_features.append(src_features[i][1:])
            tgt_scores.append(src_scores[i][1])
            
    tgt_features = np.asarray(tgt_features, dtype = np.float32)
    tgt_scores = np.asarray(tgt_scores, dtype = np.float32)
    
    return [tgt_features, tgt_scores]

def filter_NA(src):
    
    tgt = []
    for row in src:
        if not('N/A' in row):
            tgt.append(row)
    
    return tgt

def pick_columns(src, tgt_indexes):
    
    src = np.asarray(src)
    
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
    
    tgt = tgt.tolist()
    
    return tgt
    

    
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

