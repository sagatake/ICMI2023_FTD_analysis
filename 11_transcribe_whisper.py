#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:16:19 2023

@author: takeshi-s

https://aiacademy.jp/media/?p=3512
https://github.com/openai/whisper
https://pytorch.org/get-started/previous-versions/

Although I didn't customize whisper codes, there are several tips you can tune (e.g. processing interval)
(Japanese) https://qiita.com/halhorn/items/d2672eee452ba5eb6241

additional article about Whisper
(Japanese) https://ysdyt.hatenablog.jp/entry/whisper

Average: (8.02 audio sec.)/(1 process sec.)

"""
from matplotlib import pyplot as plt
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

import whisper
from mutagen.mp3 import MP3
import pathlib

TQDM = True

def main():
    "Main function"
    
    print('Loading model...', end='', flush=True)
    # model =whisper.load_model("tiny")
    # model = whisper.load_model("base")
    # model = whisper.load_model("medium")
    # model = whisper.load_model("large")
    model = whisper.load_model("large-v2")
    print('Done')
    
    # input_dir = 'dataset/data_audio/'
    input_dir = 'dataset/data_audio_separated180/'
    input_dir = pathlib.Path(input_dir)
    
    # output_dir = 'dataset/data_transcript/'
    output_dir = 'dataset/data_transcript_separated180/'
    output_dir = pathlib.Path(output_dir)
    
    time_list = [['audio_duration', 'process_duration', 'process audio duration per sec (audio_duration/process_duration)']]
    record_path = 'log_transcribe_duration.csv'
    
    dirs = [x for x in input_dir.iterdir()]
    
    for audio_path in tqdm(dirs, disable=not(TQDM)):
        
        try:
            
            _ = audio_path.with_suffix('.txt')
            text_path = output_dir / _.name
            
            audio = MP3(audio_path)
            audio_duration = audio.info.length
            
            s_time = time.time()
            result = model.transcribe(str(audio_path))
            e_time = time.time()
            process_duration = e_time - s_time

            text_segment_list = []
            for segment in result['segments']:
                text_segment = segment["text"]
                # print(segment_text)
                text_segment_list.append(text_segment)
            text = '\n'.join(text_segment_list)    
            
            with open(text_path, 'w') as f:
                f.write(text)
                
            time_list.append([audio_duration, process_duration, audio_duration/process_duration])
            with open(record_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(time_list)
            
            if not(TQDM):
                
                print(audio_path)
                print(text_path)
                print(audio_duration)            
                print('processing time: {:.3f} for {:.3f} audio file [sec]'.format(process_duration, audio_duration))            
                print(text)
            
        except Exception as e:
            print(e)
            

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

