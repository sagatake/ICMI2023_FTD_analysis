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

from pydub import AudioSegment

def main():
    "Main function"
    
    src_dir = Path('dataset/data_audio/')
    tgt_dir = Path('dataset/data_audio_separated180/')
    
    audio_files = []
    for x in src_dir.iterdir():
        for key in ['_7.mp3', '_8.mp3', '_9.mp3']:
            if key in str(x):
                audio_files.append(x)
    audio_files = sorted(audio_files)
    # pp.pprint(audio_files)
    
    duration_list = []
    
    for i, src_file in enumerate(tqdm(audio_files)):
        
        try:
            
            audio = AudioSegment.from_mp3(src_file)
            duration_3_3 = len(audio)
            duration_1_3 = math.ceil(duration_3_3 / 3 * 1)
            duration_2_3 = math.ceil(duration_3_3 / 3 * 2)
            
            print()
            print('Duration: {:05.1f} / {:05.1f} / {:05.1f} sec. for {}'.format(
                duration_1_3/1000, duration_2_3/1000, duration_3_3/1000, src_file.name))
            
            suffix = src_file.suffix
            
            tgt_file = src_file.stem + '_1' + suffix
            tgt_file = tgt_dir / tgt_file
            tgt_audio = audio[:duration_1_3]
            tgt_audio.export(tgt_file, format='mp3')
    
            tgt_file = src_file.stem + '_2' + suffix
            tgt_file = tgt_dir / tgt_file
            tgt_audio = audio[duration_1_3:duration_2_3]
            tgt_audio.export(tgt_file, format='mp3')
            
            tgt_file = src_file.stem + '_3' + suffix
            tgt_file = tgt_dir / tgt_file
            tgt_audio = audio[duration_2_3:]
            tgt_audio.export(tgt_file, format='mp3')
            
            duration_list.append([duration_3_3/1000])
            
            # if i == 2:
            #     break
        
        except Exception as e:
            
            print('Error in ', src_file.name)
            print(traceback.format_exc())
    
    min_val = np.amin(duration_list, axis=0)[0]
    ave_val = np.average(duration_list, axis=0)[0]
    max_val = np.amax(duration_list, axis=0)[0]
    
    # print(min_val)
    # print(ave_val)
    # print(max_val)
    
    print('min {:.1f}, ave {:.1f}, max {:.1f}'.format(min_val, ave_val, max_val))
    _write_csv('audio180_durations.csv', duration_list)

        
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

