## Make sure you have ffmpeg installed
import os
# import tensorflow
from os import path
import glob
import json
import wave
import struct

if not os.path.isdir("./processed_dataset"):
    os.makedirs("./processed_dataset")

def wav_to_floats(wave_file):
    w = wave.open(wave_file)
    astr = w.readframes(w.getnframes())
    # convert binary chunks to short 
    a = struct.unpack("%ih" % (w.getnframes()* w.getnchannels()), astr)
    a = [val for val in a]
    # a = [float(val) / pow(2, 15) for val in a]
    return a

# search_path = os.path.join('P4_recordings_JT', '*.ogg')
# for ogg_path in glob.glob(search_path):
#     file_name = ogg_path.split('/')[1]
#     word = file_name.split('_')[0]
#     word_id = file_name.split('_')[1].split('.')[0]
#     word = word.lower()

#     output_path = f'processed_dataset/{word}'
#     if not os.path.isdir(output_path):
#         os.makedirs(output_path)
#     os.system(f"ffmpeg -i {ogg_path} -ar 16000 -acodec pcm_s16le processed_dataset/{word}/{word_id}_nohash_0.wav")
#     data = {}
#     signal = wav_to_floats(f"./processed_dataset/{word}/{word_id}_nohash_0.wav")
#     data["values"] = signal[4000:20000]
#     output_path = f'processed_json/{word}'
#     if not os.path.isdir(output_path):
#         os.makedirs(output_path)
#     with open(f'./processed_json/{word}/{word_id}_nohash_0.json', 'w') as f:
#         json.dump(data, f)

# search_path = os.path.join('P4_recording2_rc', '*.ogg')
# counter = 0
# for ogg_path in glob.glob(search_path):
#     file_name = ogg_path.split('/')[1]
#     # print(file_name)
#     word = file_name.split('_')[0]
#     counter += 1
#     word_id = "rc{}".format(counter)
#     word = word.lower()

#     output_path = f'processed_dataset/{word}'
#     if not os.path.isdir(output_path):
#         os.makedirs(output_path)
#     ogg_path = ogg_path.replace(" ", "/ ")
#     os.system(f"ffmpeg -i {ogg_path} -ar 16000 processed_dataset/{word}/{word_id}_nohash_0.wav")

# search_path = os.path.join('p4_recording_rck2', '*.ogg')
# counter = 0
# for ogg_path in glob.glob(search_path):
#     file_name = ogg_path.split('/')[1]
#     # print(file_name)
#     word = file_name.split('_')[0]
#     counter += 1
#     word_id = "rck2{}".format(counter)
#     word = word.lower()

#     output_path = f'processed_dataset/{word}'
#     if not os.path.isdir(output_path):
#         os.makedirs(output_path)
#     ogg_path = ogg_path.replace(" ", "/ ")
#     os.system(f"ffmpeg -i {ogg_path} -ar 16000 -acodec pcm_s16le processed_dataset/{word}/{word_id}_nohash_0.wav")
#     data = {}
#     signal = wav_to_floats(f"./processed_dataset/{word}/{word_id}_nohash_0.wav")
#     data["values"] = signal[4000:20000]
#     with open(f'./processed_json/{word}/{word_id}_nohash_0.json', 'w') as f:
#         json.dump(data, f)



search_path = os.path.join('CN_digits_bk', '*.json')
counter = 0
print(len(glob.glob(search_path)))
for ogg_path in glob.glob(search_path):
    file_name = ogg_path.split('/')[1]
    # print(file_name)
    word = file_name.split('_')[0]
    # if word == "one":
    #     word = "class1"
    # if word == "ltwo":
    #     word = "class2"
    # if word == "three":
    #     word = "class3"
    # if word == "four":
    #     word = "class4"
    # if word == "five":
    #     word = "class5"
    # if word == "silence":
    #     word = "class0"
    # if word == "unknown":
    #     word = "classx"
    counter += 1
    word_id = str(counter)
    word = word.lower()
    # print(ogg_path)

    with open(ogg_path, encoding='utf-8', errors='ignore') as json_data:
        data = json.load(json_data, strict=False)
        if len(data['payload']['values']) != 16000:
            print(len(data['payload']['values']))
            print(ogg_path)
        
        # with open(f'./processed_json/{word}/{word_id}_nohash_0.json', 'w') as f:
        #     json.dump(data, f)
        # print(len(data['payload']['values']))
    # with open(str(ogg_path)) as f:
    #     print(f)
    #     data = json.loads(f.read().decode("utf-8"))
        
    #     print(len(data['payload']['values']))
    # os.rename(ogg_path,f'processed_digits/{word}_{word_id}.json') 