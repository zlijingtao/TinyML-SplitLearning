## Make sure you have ffmpeg installed
import os
# import tensorflow
from os import path
import glob
import json
import wave
import struct
import numpy as np

def wav_to_floats(wave_file):
    w = wave.open(wave_file)
    astr = w.readframes(w.getnframes())
    # convert binary chunks to short 
    a = struct.unpack("%ih" % (w.getnframes()* w.getnchannels()), astr)
    a = [val for val in a]
    # a = [float(val) / pow(2, 15) for val in a]
    return a

#delete #>500 samples under CN_digts

word_name_list = ['_background_noise_']
for word in word_name_list:
    search_path = os.path.join(f'dataset/{word}', '*.wav')
    counter = 0
    while counter < 1000:
        for ogg_path in glob.glob(search_path):

            file_name = ogg_path.split('/')[-1]
            output_path = f'datasets/EN_digits'
            data = {"payload":{}}
            signal = wav_to_floats(ogg_path)
            if len(signal) == 16000:
                data["payload"]["values"] = signal
                counter += 1
                with open(f'{output_path}/{word}.{counter}.json', 'w') as f:
                    json.dump(data, f)
            else: #background noise
                rand_idx = np.random.random_integers(0, 50)
                data["payload"]["values"] = signal[rand_idx*16000:(rand_idx+1)*16000]
                counter += 1
                with open(f'{output_path}/silence.{500+counter}.json', 'w') as f:
                    json.dump(data, f)
            if counter == 1000:
                break


word_name_list = ['no', 'off', 'marvin', 'follow', 'cat', 'house', 'learn', 'sheila', 'visual', 'zero']
counter = 0
while counter < 1000:
    for word in word_name_list:
        search_path = os.path.join(f'dataset/{word}', '*.wav')
        
        ogg_path = glob.glob(search_path)[counter]
        output_path = f'datasets/EN_digits'
        data = {"payload":{}}
        signal = wav_to_floats(ogg_path)
        if len(signal) == 16000:
            data["payload"]["values"] = signal
            counter += 1
            print(f"add {ogg_path}")
            with open(f'{output_path}/unknown.{500+counter}.json', 'w') as f:
                json.dump(data, f)
        if counter == 1000:
            break

word_name_list = ['one', 'two', 'three', 'four', 'five']
for word in word_name_list:
    search_path = os.path.join(f'dataset/{word}', '*.wav')
    counter = 0
    for ogg_path in glob.glob(search_path):
        
        file_name = ogg_path.split('/')[-1]
        output_path = f'datasets/EN_digits'
        data = {"payload":{}}
        signal = wav_to_floats(ogg_path)
        print(len(signal))
        if len(signal) == 16000:
            data["payload"]["values"] = signal
            counter += 1
            with open(f'{output_path}/{word}.{500+counter}.json', 'w') as f:
                json.dump(data, f)
        if counter == 1000:
            break
