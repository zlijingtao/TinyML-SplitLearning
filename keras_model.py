import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed, ReLU
from tensorflow.keras.optimizers import Adam, SGD
import random
from models import server_conv2d_model, server_conv_model, server_model, client_conv2d_model, client_conv_model, client_model
import numpy as np
import torch
import json
import os
import torch.nn as nn
from third_party_package import speechpy
experiment = "EN_digits"
DEBUG = False
momentum = 0.6
learningRate= 0.005
number_hidden = 1
hidden_size = 256

samples_per_device = 6300 # Amount of samples of each word to send to each device
size_output_nodes = 7
batch_size = 14 # Must be even, hsa to be split into 2 types of samples
total_samples = 7000


digits_silence_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("silence") and int(file.split(".")[1])>=500]
digits_one_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("one") and int(file.split(".")[1])>=500]
digits_two_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("two") and int(file.split(".")[1])>=500]
digits_three_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("three") and int(file.split(".")[1])>=500]
digits_four_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("four") and int(file.split(".")[1])>=500]
digits_five_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("five") and int(file.split(".")[1])>=500]
digits_unknown_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("unknown") and int(file.split(".")[1])>=500]


random.shuffle(digits_silence_files)
random.shuffle(digits_one_files)
random.shuffle(digits_two_files)
random.shuffle(digits_three_files)
random.shuffle(digits_four_files)
random.shuffle(digits_five_files)
random.shuffle(digits_unknown_files)


def raw_to_mfcc(filename):
    with open(f'./datasets/{experiment}/'+filename) as f:
        data = json.load(f)
        if 'payload' in data:
            raw_dt_npy = np.array(data['payload']['values'],dtype=float)
        else:
            raw_dt_npy = np.array(data['values'],dtype=float)
    raw_dt_npy=speechpy.processing.preemphasis(raw_dt_npy,cof=0.98, shift=1)
    mfccs = speechpy.feature.mfcc(raw_dt_npy, frame_stride=0.020,frame_length=0.020,sampling_frequency=16000, implementation_version = 2, fft_length=256, low_frequency=0, num_filters=32)
    mfcc_cmvn = speechpy.processing.cmvnw(mfccs, win_size=101, variance_normalization=True).flatten()
    
    return mfcc_cmvn

def getSamplesIIDDigits(batch_size, batch_start_index):
    global digits_silence_files, digits_one_files, digits_two_files, digits_three_files, digits_four_files, digits_five_files, digits_unknown_files

    # each_sample_amt = int(batch_size/2)
    
    start = batch_start_index
    end = batch_start_index + batch_size
    real_start = start // 7
    real_end = (end - start) // 7 + start // 7
    
    input_list = []
    label_list = []
    
    for i in range(real_start, real_end):
        
        filename = digits_silence_files[i]
        num_button = 1
        try: 
            input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        except:
            input_array = raw_to_mfcc(filename)
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.
        
        filename = digits_one_files[i]
        num_button = 2
        try: 
            input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        except:
            input_array = raw_to_mfcc(filename)
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_two_files[i]
        num_button = 3
        try: 
            input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        except:
            input_array = raw_to_mfcc(filename)
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_three_files[i]
        num_button = 4
        try: 
            input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        except:
            input_array = raw_to_mfcc(filename)
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_four_files[i]
        num_button = 5
        try: 
            input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        except:
            input_array = raw_to_mfcc(filename)
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_five_files[i]
        num_button = 6
        try: 
            input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        except:
            input_array = raw_to_mfcc(filename)
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_unknown_files[i]
        num_button = 7
        try: 
            input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        except:
            input_array = raw_to_mfcc(filename)
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

    input_list_array = np.concatenate(input_list, axis = 0)
    label_list_array = np.array(label_list).reshape(-1, 1)
    return input_list_array, label_list_array



train_in, train_out = getSamplesIIDDigits(samples_per_device, 0)
test_in, test_out = getSamplesIIDDigits(total_samples - samples_per_device, samples_per_device)

train_in = np.reshape(train_in, (samples_per_device, 650))
test_in = np.reshape(test_in, (total_samples - samples_per_device, 650))


save_file_name = "large_conv2d_model_momentum0.6"

c_path = f"./saved_results/EN_digits/{save_file_name}/c_model.pt"
s_path = f"./saved_results/EN_digits/{save_file_name}/s_model.pt"

# create the whole model
class Mywhole_model(nn.Module):
    def __init__(self, model_c, model_s):
        super(Mywhole_model, self).__init__()
        self.model_c=model_c
        self.model_s=model_s

    def forward(self,x):
        x=self.model_c(x)
        x=self.model_s(x)
        return x
## if c_model and s_model are loaded from saved models: 
modelC= client_conv2d_model()
modelS= server_conv2d_model(num_class = size_output_nodes, number_hidden = number_hidden, hidden_size = hidden_size)

modelC.load_state_dict(torch.load(c_path))
modelS.load_state_dict(torch.load(s_path))

whole_model=Mywhole_model(modelC, modelS)



# whole model validate function
def whole_validate(test_in, test_out):
    # multiple_batch
    input = torch.tensor(test_in).float()
    
    label = torch.from_numpy(test_out).view(input.size(0),).long()

    whole_model.eval()
    with torch.no_grad():
        output = whole_model(input)
        
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, label)

    error = loss.detach().numpy() / input.size(0)

    accu = (torch.argmax(output, dim = 1) == label).sum() / input.size(0)

    accu = accu.detach().numpy()
    print("pytorch model's accuracy is:", accu)



whole_validate(test_in, test_out)


# model architecture
model = Sequential()
classes = 7
channels = 1
columns = 13
rows = int(650 / (columns * channels))
# model.add(Reshape((rows, columns, channels), input_shape=(650, )))
# model.add(Reshape((channels, rows, columns), input_shape=(650, )))
model.add(Reshape((columns, rows, channels), input_shape=(650, )))
# model.add(Conv2D(12, kernel_size=3, strides=2, activation='relu', padding='valid', use_bias = False))
model.add(Conv2D(12, kernel_size=3, strides=2, activation='relu', padding='valid', use_bias = False))
# model.add(Dropout(0.5))
# model.add(Conv2D(30, kernel_size=3, strides=2, activation=None, padding='valid', use_bias = False))
model.add(Conv2D(30, kernel_size=3, strides=2, activation=None, padding='valid', use_bias = False))
model.add(BatchNormalization(epsilon=1e-05, momentum=0.1)) # need to change?
model.add(ReLU())
# model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(classes, activation='softmax',name='y_logis'))

# this controls the learning rate
opt = SGD(learning_rate=0.005, momentum= 0.6)
# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
BATCH_SIZE = 1


torch_weight = []
bias_param = []
first_encounter = True
batchnorm_size = 30
for param in whole_model.parameters():
    if len(param.data.size()) == 4:
        torch_weight.append([param.data.numpy().transpose(2, 3, 1, 0)])
    elif len(param.data.size()) == 2:
        if first_encounter:
            bias_param.append(np.transpose(param.data.numpy()).reshape(30,2,11,256).transpose(1, 2, 0, 3).reshape(660, 256))
            first_encounter = False
        else:
            bias_param.append(np.transpose(param.data.numpy()))
    elif len(param.data.size()) == 1:
        if len(bias_param) == 1:
            bias_param.append(param.data.numpy())
                
            torch_weight.append(bias_param[:])
            
            bias_param = []
        else:
            bias_param.append(param.data.numpy())
if DEBUG:
    print("torch weight:")
    for item in torch_weight:
        if isinstance(item, list):
            for i in item:
                print(i.shape)
        else:
            print(item.shape)

s_checkpoint = torch.load(s_path)
bn_weight = []
for key in s_checkpoint.keys():
    if 'server.1' in key and ('num_batches_tracked' not in key):
        print(key)
        bn_weight.append(s_checkpoint[key].numpy())
torch_weight[2] = bn_weight



count = 0
for i, layer in enumerate(model.layers):
    if len(layer.get_weights()) > 0:
        print(len(layer.get_weights()))
        layer.set_weights(torch_weight[count])
        # layer.kernel.assign(torch_weight[count])
        count += 1



# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


b = np.zeros((test_out.flatten().size, test_out.max()+1))
b[np.arange(test_out.flatten().size),test_out.flatten()] = 1

if DEBUG:
    print("Weights and biases of the layers before training the model: \n")
    for layer in model.layers:
        print(layer.name)
        if (len(layer.get_weights()) == 1):
            print("Weights")
            print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
        if (len(layer.get_weights()) == 2):
            print("Weights")
            print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
            print("Bias")
            print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')
        if (len(layer.get_weights()) == 4):
            print("BN1")
            print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
            print("BN2")
            print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1])
            print("BN3")
            print("Shape: ",layer.get_weights()[2].shape,'\n',layer.get_weights()[2])
            print("BN4")
            print("Shape: ",layer.get_weights()[3].shape,'\n',layer.get_weights()[3],'\n')

results = model.evaluate(test_in, b, batch_size=1)
print("test loss, test acc:", results)

# Use this flag to disable per-channel quantization for a model.
# This can reduce RAM usage for convolutional models, but may have
# an impact on accuracy.

if DEBUG:
    from functools import reduce
    from typing import Union

    def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                        access_string: str):
        """Retrieve a module nested in another by its access string.

        Works even when there is a Sequential in the module.
        """
        names = access_string.split(sep='.')
        return reduce(getattr, names, module)


    for name, layer in modelS.named_modules():
        print(name, layer)

    conv2 = get_module_by_name(modelS, 'server.0')
    bn1 = get_module_by_name(modelS, 'server.1')
    relu1 = get_module_by_name(modelS, 'server.2')
    flatten = get_module_by_name(modelS, 'server.3')
    fc1 = get_module_by_name(modelS, 'server.4')
    relu2 = get_module_by_name(modelS, 'server.5')

    output = modelC(torch.tensor(test_in[:1,:]).float())
    output = conv2(output)
    output = bn1(output)
    output = relu1(output)
    output = flatten(output)
    output = fc1(output)
    output = relu2(output)
    print(output.size())
    print(output[0, :])

    output = model.layers[0](test_in[:1,:])
    output = model.layers[1](output)
    output = model.layers[2](output)
    output = model.layers[3](output)
    output = model.layers[4](output)
    output = model.layers[5](output)
    output = model.layers[6](output)
    print(output.shape)
    print(output[0, :])
    print(modelS)

# Implement keras to tensorflow lite

## get representative data for quant
def rep_dataset(): #test_in is global #FIXME change to tensor?
    test_data = np.reshape(test_in, (total_samples - samples_per_device, 1, 13, 50))
    for input_value in tf.data.Dataset.from_tensor_slices(test_data).batch(1).take(100):
        yield [input_value]

# converter = tf.lite.TFLiteConverter.from_saved_model(f"{saved_dir}/{transfered_model}.pb")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# --------- add quant here ------------
# Convert using float fallback quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
# #using integer-only quantization #TODO
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8

# write to tflite model
tflite_model = converter.convert() #TODO: add quant
tfl_model_path = f"./saved_results/EN_digits/{save_file_name}/whole_model.tflite"
open(tfl_model_path, "wb").write(tflite_model)
print("------ finished writing tflite model ------------")
# print(f"tlf_model saved to: {saved_dir}/{transfered_model}.tflite")

## evaluate tflite model
## --------------- evaluate pytorch and tfl model ---------------
# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_out_indices):
    global test_out
    global test_in

    # process test out
    input = torch.tensor(test_in).float()
    label = (torch.from_numpy(test_out).view(input.size(0),).long()).numpy()
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_out_indices),), dtype=int)

    for i, test_image_index in enumerate(test_out_indices):
        test_image = input[test_image_index]
        test_label = label[test_image_index]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        predictions[i] = np.argmax(output)
    return predictions

# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file):
    global test_in
    global test_out
    input = torch.tensor(test_in).float()
    label = (torch.from_numpy(test_out).view(input.size(0),).long()).numpy()
    test_out_indices = range(test_in.shape[0])
    predictions = run_tflite_model(tflite_file, test_out_indices)
    accuracy = np.sum((predictions== label)) / len(test_in)

    print('TFLite model accuracy is %.4f (Number of test samples=%d)' % (
        accuracy, len(test_in)))

## ---------- eva ----------
TF_model = tfl_model_path # specific file location
evaluate_model(TF_model)