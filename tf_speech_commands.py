# %%

# conda create -n cen598 python=3.8

# pip install tensorflow

# pip install -q tfds-nightly

# conda install numpy

WANTED_WORDS = "backward,down,up,forward,left,right"
TRAINING_STEPS = "6000,6000,3000"
# LEARNING_RATE = "0.001,0.0001"
LEARNING_RATE = "0.005,0.0005,0.0001"
# Calculate the total number of steps, which is used to identify the checkpoint
# file name.
TOTAL_STEPS = str(sum(map(lambda string: int(string), TRAINING_STEPS.split(","))))

# Print the configuration to confirm it
print("Training these words: %s" % WANTED_WORDS)
print("Training steps in each stage: %s" % TRAINING_STEPS)
print("Learning rate in each stage: %s" % LEARNING_RATE)
print("Total number of training steps: %s" % TOTAL_STEPS)

# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2 # for 'silence' and 'unknown' label
equal_percentage_of_training_samples = int(100.0/(number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants which are shared during training and inference
PREPROCESS = 'micro'
WINDOW_STRIDE = 20
MODEL_ARCHITECTURE = 'tiny_conv' # Other options include: single_fc, conv,
                      # low_latency_conv, low_latency_svdf, tiny_embedding_conv

# Constants used during training only
VERBOSITY = 'WARN'
EVAL_STEP_INTERVAL = '1000'
SAVE_STEP_INTERVAL = '1000'

# Constants for training directories and filepaths
DATASET_DIR =  'dataset/'
LOGS_DIR = 'logs/'
TRAIN_DIR = 'train/' # for training checkpoints and other files.

# %%
# Constants for inference directories and filepaths
import os
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
  os.mkdir(MODELS_DIR)
MODEL_TF = os.path.join(MODELS_DIR, 'model.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, 'model.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_model.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'model.cc')
SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')

QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
QUANT_INPUT_RANGE = QUANT_INPUT_MAX - QUANT_INPUT_MIN

import tensorflow as tf

import subprocess

os.system("git clone -q --depth 1 https://github.com/tensorflow/tensorflow")

os.system("cp -R processed_dataset/* dataset/")

PYTHON_PATH = '/home/jingtao1/miniconda3/envs/tf115/bin/python'

subprocess.call(f"{PYTHON_PATH} ./tensorflow/tensorflow/examples/speech_commands/train.py \
--data_dir={DATASET_DIR} \
--wanted_words={WANTED_WORDS} \
--silence_percentage={SILENT_PERCENTAGE} \
--unknown_percentage={UNKNOWN_PERCENTAGE} \
--preprocess={PREPROCESS} \
--window_stride={WINDOW_STRIDE} \
--model_architecture={MODEL_ARCHITECTURE} \
--how_many_training_steps={TRAINING_STEPS} \
--learning_rate={LEARNING_RATE} \
--train_dir={TRAIN_DIR} \
--summaries_dir={LOGS_DIR} \
--verbosity={VERBOSITY} \
--eval_step_interval={EVAL_STEP_INTERVAL} \
--save_step_interval={SAVE_STEP_INTERVAL} --verbosity=DEBUG", shell = True)

# %%

os.system(f"rm -rf {SAVED_MODEL}")

subprocess.call(f"{PYTHON_PATH} ./tensorflow/tensorflow/examples/speech_commands/freeze.py \
--wanted_words={WANTED_WORDS} \
--window_stride_ms={WINDOW_STRIDE} \
--preprocess={PREPROCESS} \
--model_architecture={MODEL_ARCHITECTURE} \
--start_checkpoint={TRAIN_DIR}{MODEL_ARCHITECTURE}'.ckpt-'{TOTAL_STEPS} \
--save_format=saved_model \
--output_file={SAVED_MODEL}", shell = True)

# import tensorboard

# %%

# '''TODO: Below are untested'''
import sys

# We add this path so we can import the speech processing modules.
sys.path.append("./tensorflow/tensorflow/examples/speech_commands/")
import input_data
import models
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10


model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
    SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
audio_processor = input_data.AudioProcessor(
    DATA_URL, DATASET_DIR,
    SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
    WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
    TESTING_PERCENTAGE, model_settings, LOGS_DIR)

with tf.Session() as sess:

  print("START")
  print(SAVED_MODEL)
  float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  float_tflite_model = float_converter.convert()
  float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
  print("Float model is %d bytes" % float_tflite_model_size)

  converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.inference_input_type = tf.lite.constants.INT8
  converter.inference_output_type = tf.lite.constants.INT8
  def representative_dataset_gen():
    for i in range(100):
      data, _ = audio_processor.get_data(1, i*1, model_settings,
                                        BACKGROUND_FREQUENCY, 
                                        BACKGROUND_VOLUME_RANGE,
                                        TIME_SHIFT_MS,
                                        'testing',
                                        sess)
      flattened_data = np.array(data.flatten(), dtype=np.float32)
      try:
        flattened_data = flattened_data.reshape(1, 1960)
      except:
        continue
      yield [flattened_data]
  converter.representative_dataset = representative_dataset_gen
  tflite_model = converter.convert()
  tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
  print("Quantized model is %d bytes" % tflite_model_size)


  # initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(FLOAT_MODEL_TFLITE))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  batch_size = 100
  set_size = audio_processor.set_size('testing')
  predictions = np.zeros((set_size,), dtype=int)
  ground_truth = np.zeros((set_size,), dtype=int)
  for i in range(0, set_size, 1):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        1, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    # batch_size = min(batch_size, set_size - i)
    # print(input_details)
    # if input_details['dtype'] == numpy.uint8:
    input_scale, input_zero_point = input_details["quantization"]
    # print(test_fingerprints)
    # test_fingerprints = test_fingerprints / input_scale + input_zero_point
    test_fingerprints = test_fingerprints.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_fingerprints)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    predictions[i] = output.argmax()
    ground_truth[i] = test_ground_truth
    # print(predictions)
    # print(ground_truth)

  float_accuracy = (np.sum(predictions== ground_truth) * 100) / len(ground_truth)
  
  interpreter = tf.lite.Interpreter(model_path=str(MODEL_TFLITE))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  batch_size = 100
  set_size = audio_processor.set_size('testing')
  predictions = np.zeros((set_size,), dtype=int)
  ground_truth = np.zeros((set_size,), dtype=int)
  for i in range(0, set_size, 1):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        1, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    input_scale, input_zero_point = input_details["quantization"]
    test_fingerprints = test_fingerprints / input_scale + input_zero_point
    test_fingerprints = test_fingerprints.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_fingerprints)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    predictions[i] = output.argmax()
    ground_truth[i] = test_ground_truth
    # print(predictions)
    # print(ground_truth)

  quantize_accuracy = (np.sum(predictions== ground_truth) * 100) / len(ground_truth)
  print("Quantized model accuracy is ", quantize_accuracy)
  print("Float model accuracy is ", float_accuracy)
 
# os.system("xxd -i models/model_new.tflite models/model.cpp")
