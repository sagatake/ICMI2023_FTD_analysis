#!/usr/bin/env python3
# coding: utf-8
"""
Created on Mon Mar  1 10:15:00 2022
Updated on Fri Jun 17 12:18:00 2022
Updated on Mon Mar 6 21:43:00 2023

@author: takeshi-s

It is better to use Windows machine since Openface on docker support multiprocessing only for Windows 

"""

import os
import csv
import sys
import cv2
import json
import time
import copy
import shutil
import platform
import datetime
import argparse
import traceback
import threading
import subprocess
import numpy as np
import pandas as pd
import pprint as pp
import pickle as pkl
from tqdm import tqdm
# from pydub import AudioSegment

sys.path.append(os.path.dirname(__file__))

from pathlib import Path
import librosa
import torch

import func_text_parallel as func_text

import gc
from concurrent import futures

import multiprocessing
NUM_WORKER = multiprocessing.cpu_count()
# NUM_WORKER = 8
# NUM_WORKER = 20
# NUM_WORKER = 1

DEBUG = False
ENABLE_TQDM = True

NUM_TEST = None
# NUM_TEST = NUM_WORKER

WITH_BERT = True

pf = platform.system()

results = {}
durations = []

# derailment, incoherence, tangentiality    : BERT
# illogicality                              : anaphoric analysis
# clanging                                  : phone similarity
# pressured speech                          : WPM, sentence end prediction, ave loudness

"""
IMPREMENTED:
    BERT_word
    BERT_sent
    BERT_cont
    num_content
    WPM
    CDI
    ratio_POS

NEED TO IMPREMENT:
    None
"""

global_s_time = time.time()

def main():
    
    text_dir = Path('dataset/data_transcript/')
    # text_dir = Path('dataset/data_transcript_separated180/')

    audio_dir = Path('dataset/data_audio/')
    # audio_dir = Path('dataset/data_audio_separated180/')
    
    out_file = 'features.csv'
    # out_file = 'features_separated180.csv'
    
    duration_file = 'durations.csv'
    # duration_file = 'durations_separated180.csv'

    _write_csv(duration_file, [[0.0]])
    
    texts = []
    audios = []
    
    features_list = []
    future_list = []
    
    duration = 0
    ETA = 0
    
    text_files = sorted(list(text_dir.iterdir()))
    
    with futures.ProcessPoolExecutor(max_workers = NUM_WORKER) as executor:
        
        for i, text_file in enumerate(tqdm(text_files, disable = not(ENABLE_TQDM))):
            
            
            if DEBUG:
                print('##### Test initial run #####')
                _ = calc_parallel(i, audio_dir, text_files, duration_file)
            
            future = executor.submit(calc_parallel, i, audio_dir, text_files, duration_file)
            future_list.append(future)
            
            if (NUM_TEST != None) and (NUM_TEST == (i-1)):
                break
        
        output = futures.as_completed(future_list)
        # print(output)
        # for row in output:
            # print(row)
            # print(row.result())
        output = [x.result() for x in output]
        output = [x for x in output if x != None] #eliminate failed cases
        output = sorted(output, key = lambda x:x[0][1])
        pp.pprint(output)
        feature_names = output[0][1]
        features_list = [x[0] for x in output]

    feature_names[0].insert(0, 'file_name')
    feature_names[0].insert(1, 'index')
    feature_names[1].insert(0, 'file_name')
    feature_names[1].insert(1, 'index')
    features_list.insert(0, feature_names[0])
    features_list.insert(1, feature_names[1])

    _write_csv(out_file, features_list)

def calc_parallel(i, audio_dir, text_files, duration_file):
    
    print('text {}/{} start'.format(i, len(text_files)), flush=True)

    func_text_obj = func_text.Func_text(WITH_BERT)
    
    s_time = time.time()

    text_file = text_files[i]
        
    try:
        with open(text_file, 'r') as f:
            text = f.read()
        
        name = text_file.stem + '.mp3'
        audio_file = audio_dir / name
        audio, sr = librosa.load(audio_file)
        
    except Exception as e:

        print(e)
        print(traceback.format_exc())
        return None
    
    features, feature_names = calc_text(i, text, audio, sr, func_text_obj)
    if features == None:
        return None
    # features = [], feature_names = []
    features.insert(0, text_files[i].stem)
    # features_list.append(features)
    
    if WITH_BERT:
        func_text_obj.bert_model.to('cpu')
        del func_text_obj
        torch.cuda.empty_cache()
    
    gc.collect()
    if DEBUG:
        print(gc.get_stats()[-1])
        
    e_time = time.time()
    duration = e_time - s_time

    try:
            
        durations = _load_csv(duration_file)
        durations = np.asarray(durations, dtype=np.float32).tolist()
        print('Durations dim', np.shape(durations))
        durations[0].append(duration)
        _write_csv(duration_file, durations)
        
        global_duration = e_time - global_s_time
        ETA = (global_duration / len(durations[0][1:])) * (len(text_files) - (i + 1)) / 60 / 60

    except Exception as e:
        
        print(e)
        print(traceback.format_exc())
        
        ETA = None
    
    skip_index_list = [i for i in range(NUM_WORKER-1)]
    if (i in skip_index_list) and (duration != None):
        print('text : {}/{} (last_duration: {:.1f} sec., ETA: N/A)'.format(
            i, len(text_files), duration
            ),
            flush=True
        )
    elif (duration != None) and (ETA != None):
        print('text : {}/{} (last_duration: {:.1f} sec., ETA: {:.1f} h.)'.format(
            i, len(text_files), duration, ETA
            ),
            flush=True
        )
    elif (duration != None):
        print('text : {}/{} (last_duration: {:.1f} sec., ETA: N/A)'.format(
            i, len(text_files), duration
            ),
            flush=True
        )
    else:
        print('text : {}/{} (last_duration: N/A, ETA: N/A'.format(i, len(text_files)), flush=True)
    
    result = [features, feature_names]
    
    return result

    
def calc_text(i, text, audio, sr, func_text_obj):
    
    try:
        # unit: minute
        audio_length = librosa.get_duration(y=audio, sr=sr)
        
        text_feature_names = [
            [], #Japanese
            [] #English
            ]
                    
        row = [i]
        
        feature_name_JP = 'BERT_word'
        feature_name_EN = 'BERT_word'
        feature = func_text_obj.BERT_word(text)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)
            
        feature_name_JP = 'BERT_sent_diff'
        feature_name_EN = 'BERT_sent_diff'
        feature = func_text_obj.BERT_sentence_average_diff(text)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)
        
        feature_name_JP = 'BERT_sent_cosine'
        feature_name_EN = 'BERT_sent_cosine'
        feature = func_text_obj.BERT_sentence_average_cosine(text)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)
        
        feature_name_JP = 'BERT_content'
        feature_name_EN = 'BERT_content'
        feature = func_text_obj.BERT_cont_word(text)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)
        
        #
        # Redundant: will be ignored in 30_filter_features_for_ablation.py
        #
        feature_name_JP = 'ratio_content'
        feature_name_EN = 'ratio_content'
        feature = func_text_obj.count_content_words(text)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)

        #
        # Redundant: will be ignored in 30_filter_features_for_ablation.py
        #
        feature_name_JP = 'WPM'
        feature_name_EN = 'WPM'
        feature = func_text_obj.calc_WPM(text, audio_length)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)

        feature_name_JP = 'CDI-J'
        feature_name_EN = 'CDI-J'
        feature = func_text_obj.calc_CDI_J(text)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)
    
        feature_name_JP = 'num_negation'
        feature_name_EN = 'num_negation'
        feature = func_text_obj.count_negation(text)
        feature = [feature]
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)
            
        feature_dict = func_text_obj.count_POS_sudachi(text)
        sum_freq = sum([feature_dict[key] for key in feature_dict.keys()])
        for key in feature_dict.keys():
            
            feature_name_JP = 'ratio_{}'.format(key)
            
            if key == '補助記号':
                feature_name_EN = 'ratio_{}'.format('auxiliary-symbol')
            if key == '空白':
                feature_name_EN = 'ratio_{}'.format('space')
            if key == '名詞':
                feature_name_EN = 'ratio_{}'.format('noun')
            if key == '記号':
                feature_name_EN = 'ratio_{}'.format('symbol')
            if key == '接頭辞':
                feature_name_EN = 'ratio_{}'.format('prefix')
            if key == '感動詞':
                feature_name_EN = 'ratio_{}'.format('interjection')
            if key == '副詞':
                feature_name_EN = 'ratio_{}'.format('adverb')
            if key == '接尾辞':
                feature_name_EN = 'ratio_{}'.format('suffix')
            if key == '代名詞':
                feature_name_EN = 'ratio_{}'.format('pronoun')
            if key == '形状詞':
                feature_name_EN = 'ratio_{}'.format('adjectival-verb')
            if key == '動詞':
                feature_name_EN = 'ratio_{}'.format('verb')
            if key == '形容詞':
                feature_name_EN = 'ratio_{}'.format('adjective')
            if key == '連体詞':
                feature_name_EN = 'ratio_{}'.format('adnominal')
            if key == '接続詞':
                feature_name_EN = 'ratio_{}'.format('conjunction')
            if key == '助詞':
                feature_name_EN = 'ratio_{}'.format('particle')
            if key == '助動詞':
                feature_name_EN = 'ratio_{}'.format('auxiliary-verb')                    
                
            feature = [feature_dict[key]/sum_freq]
            row.extend(feature)
            text_feature_names[0].append(feature_name_JP)
            text_feature_names[1].append(feature_name_EN)
            if DEBUG:
                print(feature_name_JP, feature_name_EN, feature)

        feature_name_JP = 'ratio_content'
        feature_name_EN = 'ratio_content'
        feature = func_text_obj.count_content_words(text)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)
        
        feature_name_JP = 'ratio_punctuation'
        feature_name_EN = 'ratio_punctuation'
        feature = func_text_obj.count_punctuations(text)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)

        feature_name_JP = 'WPM'
        feature_name_EN = 'WPM'
        feature = func_text_obj.calc_WPM(text, audio_length)
        row.extend(feature)
        text_feature_names[0].append(feature_name_JP)
        text_feature_names[1].append(feature_name_EN)
        if DEBUG:
            print(feature_name_JP, feature_name_EN, feature)
            
        # feature_name = 'CDI'
        # feature = func_text_obj.calc_CDI(text)
        # row.extend(feature)
        # text_feature_names.append(feature_name)
        # if DEBUG:
        #     print(feature_name, feature)

        
    
        if DEBUG:
            print(row)
        
        return [row, text_feature_names]
    
    except Exception as e:
        print(traceback.format_exc())
        
        return [None, None]
                

        


def _feature_loader(csv_file_name):
    features = pd.read_csv(csv_file_name)
    feature_label = features.columns[1:]
    features = features.drop("ID", axis=1).values.tolist()
    return features, feature_label

def _add_delimiter(src):
    tgt = []
    for line in src:
        line = line + '\n'
        tgt.append(line)
    return tgt

class _Queue:          
          
    def __init__(self):
        self.queue = []
        
    def put(self, item):
        self.queue.append(item)
        
    def get(self):
        out = self.queue[0]
        self.queue = self.queue[1:]
        return out
    
    def update(self, item):
        self.put(item)
        _ = self.get()
    
    def minimum(self):
        return min(self.queue)
    
    def maximum(self):
        return max(self.queue)

def _write_csv(file_path, data, add=True):
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)

def _load_csv(input_file, delimiter = ',', encoding = 'utf-8', cast = None):
    
    with open(input_file, encoding = encoding) as f:
        reader = csv.reader(f, delimiter = delimiter)
        if cast != None:
            data = []
            for row in reader:
                temp = []
                for x in row:
                    temp.append(cast(x))
                data.append(temp)
        else:
            data = [x for x in reader]
    
    return data

def _load_csv_multiple(input_dir, delimiter = ',', keyword = None, encoding = 'utf-8', cast = None):
    
    input_files = os.listdir(input_dir)
    IDs = []
    data = []
    
    time.sleep(1)
    
    for i, input_file in enumerate(tqdm(sorted(input_files))):
                        
        if keyword != None:
            if not (keyword in input_file):
                continue

        #if DEBUG or LOCAL_DEBUG:
        #    print(input_file)
        
        ID, ext = input_file.split('.')
        IDs.append(ID)

        path = os.path.join(input_dir, input_file)
        if cast != None:
            data.append(_load_csv(path, delimiter = delimiter, encoding = encoding, cast = cast))
        else:
            data.append(_load_csv(path, delimiter = delimiter, encoding = encoding))
                        
    return [IDs, data]

def _load_text_data(input_dir, keyword = None):
    
    print('Text : Load data : Processing')
    
    output = []
    
    input_files = os.listdir(input_dir)
    IDs = []
    
    time.sleep(1)
    
    for i, input_file in enumerate(tqdm(sorted(input_files))):
        
        if keyword != None:
            if not (keyword in input_file):
                continue
        
        with open(os.path.join(input_dir, input_file)) as f:
            output_individual = []
            for line in f.readlines():
                line = line.replace('\n', '')
                output_individual.append(line)
        
        ID, ext = input_file.split('.')
        IDs.append(ID)

        output.append(output_individual)        


    print('Text : Load data : Done')
    
    return [IDs, output]
            

def _load_audio_int_data(input_dir, keyword = None):

    print('Audio : Load data : Processing')
    
    #with open(os.path.join(feature_dir, ' '+base_name+'_intensity.csv'), 'r') as f:
    #    reader = csv.reader(f, delimiter=' ')
    
    input_files = os.listdir(input_dir)
    IDs = []
    output_list = []
    
    time.sleep(1)
    
    for i, input_file in enumerate(tqdm(sorted(input_files))):
        
        if not('intensity' in input_file):
            continue
        #print(input_file)
        
        if keyword != None:
            if not (keyword in input_file):
                continue

        base, ext = input_file.split('.')
        parts = base.split('_')
        ID = '_'.join(parts[:-1])
        IDs.append(ID)
        
        data = _load_csv(os.path.join(input_dir, input_file))
    
        temp0 = []
        for y in data:
            temp1 = []
            for z in y:
                if not z=='':
                    temp1.append(z.split())
                    temp0.append(temp1)
        #pp.pprint(temp0[0:50])
        
        individual_output = []
        for i in range(len(temp0)):
            if 'z' in temp0[i][0]:
                if len(temp0[i][0])==5:
                    #print(temp0[i][0])
                    frame_num = temp0[i][0][2]
                    frame_num = frame_num[1:-2]
                    
                    #To avoid exception
                    if frame_num == '':
                        continue
                    
                    individual_output.append([int(frame_num),float(temp0[i][0][4])])
        
        output_list.append(individual_output)
        
    #data = _load_csv_multiple(input_dir, keyword = 'intensity')

    print('Audio : Load data : Done')

    return [IDs, output_list]

def _load_audio_f0_data(input_dir, keyword = None):

    print('Audio : Load data : Processing')
    
    #with open(os.path.join(feature_dir, ' '+base_name+'_intensity.csv'), 'r') as f:
    #    reader = csv.reader(f, delimiter=' ')
    
    input_files = os.listdir(input_dir)
    IDs = []
    output_list = []
    
    time.sleep(1)
    
    for i, input_file in enumerate(tqdm(sorted(input_files))):

        #print(input_file)
        
        if not('pitch' in input_file):
            continue
        
        if keyword != None:
            if not (keyword in input_file):
                continue

        base, ext = input_file.split('.')
        parts = base.split('_')
        ID = '_'.join(parts[:-1])
        IDs.append(ID)
        
        data = _load_csv(os.path.join(input_dir, input_file))
    
        data = data[11:]
        frames = []
        frame = []
        for i in range(len(data)):
            #print(data[i])
            if 'frames [' in data[i][0]:
                frames.append(frame)
                frame = [data[i][0].replace(' ', '')]
            else:
                frame.append(data[i][0].replace(' ', ''))
        frames.pop(0)
        #pp.pprint(frames[:3])
        
        frame_candidates = []
        for frame_src in frames:
            frame_tgt = []
            for cand_index in range(4, len(frame_src), 3):
                #print(frame[cand_index])
                freq = frame_src[cand_index+1]
                freq = freq.replace('frequency=','')
                strength = frame_src[cand_index+2]
                strength = strength.replace('strength=','')
                frame_tgt.append([float(freq), float(strength)])
            frame_candidates.append(frame_tgt)
            
        #pp.pprint(frame_candidates[:3])

        individual_output = []
        for i, candidates in enumerate(frame_candidates):
            max_index = np.argmax(candidates, axis=0)
            #print(max_index)
            selected = candidates[max_index[1]]
            #print(selected)
            if selected[0] == 0.0:
                continue
            individual_output.append([i+1, selected[0]])
                        
        #pp.pprint(individual_output)
        
        output_list.append(individual_output)
                
    #data = _load_csv_multiple(input_dir, keyword = 'intensity')

    print('Audio : Load data : Done')

    return [IDs, output_list]


def _load_face_data(input_dir, keyword = None):

    print('Face : Load data : Processing')
    IDs, data = _load_csv_multiple(input_dir, keyword = keyword)
    print('Face : Load data : Done')

    return [IDs, data]

def _load_body_data(input_dir, keyword = None):

    print('Body : Load data : Processing')
    IDs, data = _load_csv_multiple(input_dir, keyword = keyword, cast = float)
    print('Body : Load data : Done')

    return [IDs, data]

def _filter_by_ID(ID_list, features):

    output_list = []
    for tgt_ID in ID_list:
        tgt_idx = -1
        for i in range(len(features)):
            if features[i][0] == tgt_ID:
                tgt_idx = i
                if DEBUG:
                    print(tgt_ID)
                    print(features[i][0])
        if tgt_idx != -1:
            output_list.append(features[tgt_idx])
    
    return output_list

def _pick_correspond_subject(main_ID, inter_IDs, inter_subjects, main_keyword, inter_keyword):
    
    if DEBUG:
        print(main_ID)
    
    tgt_ID = None
    tgt_subject = None
    success = False
    
    key_ID = main_ID.replace(main_keyword, inter_keyword)
    
    for inter_ID, inter_subject in zip(inter_IDs, inter_subjects):
        if DEBUG:
            print(main_ID, inter_ID)
        if inter_ID == key_ID:
            tgt_ID = inter_ID
            tgt_subject = inter_subject
            success = True
            break
    
    return [tgt_ID, tgt_subject, success]

def _check_dir(dir_path):
    
    if not(os.path.exists(dir_path)):
        os.mkdir(dir_path)

def _find_model_path(model_names, task_name, label_name=None):
    
    if label_name == None:
        for model_name in model_names:
            if ('normalizer' in model_name) and (task_name in model_name):
                return model_name
    else:
        for model_name in model_names:
            if ('predicter' in model_name) and (task_name in model_name) and (label_name in model_name):
                return model_name
    
    print('Invalid argument: _find_model_path(model_names, task_name, label_name=None)')
    sys.exit()


def _write_json(data, name, encoding='shift-jis'):
    
    with open(name, 'w', encoding = encoding) as f:
        text = json.dumps(data, ensure_ascii = False, indent=2)
        f.write(text)
        
def _make_dirs(dir_list):
    
    for target_dir in dir_list:
        os.mkdir(target_dir)
        
def _copyfile(src, tgt):
    print('Copy {} : {} ...'.format(src, tgt))
    shutil.copyfile(src, tgt)
    print('Copy {} : {} ... Done'.format(src, tgt))
    
def _copytree(src, tgt):
    print('Copy {} : {} ...'.format(src, tgt))
    shutil.copytree(src, tgt)
    print('Copy {} : {} ... Done'.format(src, tgt))

def _add_name_tag(tgt_dir, tag):
    
    for name in os.listdir(tgt_dir):
        src_path = os.path.join(tgt_dir, name)
        base, ext = name.split('.')
        name = '{}_{}.{}'.format(base, tag, ext)
        tgt_path = os.path.join(tgt_dir, name)
        shutil.move(src_path, tgt_path)

def _remove_name_tag(tgt_dir, tag):
    
    for name in os.listdir(tgt_dir):
        src_path = os.path.join(tgt_dir, name)
        base, ext = name.split('.')
        base = base.strip('_{}'.format(tag))
        name = '{}.{}'.format(base, ext)
        tgt_path = os.path.join(tgt_dir, name)
        shutil.move(src_path, tgt_path)
        

def _replace_path_in_code(src_name, tgt_name, key, content):
        
    data = []
    with open(src_name) as f:
        for line in f.readlines():
            data.append(line)
    
    for i in range(len(data)):
        
        if key in data[i]:
            
            
            before = data[i+1]
            
            data[i+1] = content
            
            after = content
            
            if DEBUG:
                print('Before: ', before)
                print('After : ', after)
                    
    with open(tgt_name, 'w') as f:
        for line in data:
            f.write(line)
                

if __name__ == '__main__':
    
    main()
