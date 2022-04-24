import serial
from serial.tools.list_ports import comports

import struct
import time

import matplotlib.pyplot as plt
import threading
import time
import json
import os
import random

import logging
import sys

random_seed=1234
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO, console_out = True):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(handler)
    if console_out:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)
    return logger



#TODO: compare baseline validation and more layers (or larger model)
#TODO: try to extinct resources on board. (multi-iamge batch?)
#TODO: find a way to store some image and pull out them to train.
#TODO: Use our own image. 
#TODO: How to make it really useful? Just as the original one, need some button to do the labeling.

model_log_file = "./log.txt"
logger = setup_logger('main_logger', model_log_file, level=logging.DEBUG)


batch_size = 10 # Must be even, hsa to be split into 2 types of samples


running_batch_accu = 0
running_batch_accu_list = []


epoch_size = 3 # default = 1
step_size = 1 # The real batch size
experiment = 'digits' # 'iid', 'no-iid', 'train-test', 'custom', 'digits'
# model_type = "fc"
model_type = "fc"

momentum = 0.6
learningRate= 0.01
number_hidden = 0
hidden_size = 128
# initialize client-side model
size_hidden_nodes = 25
if experiment == "custom":
    size_output_nodes = 5
    samples_per_device = 250 # Amount of samples of each word to send to each device
    total_samples = 250
elif experiment == 'digits':
    samples_per_device = 315 # Amount of samples of each word to send to each device
    size_output_nodes = 7
    batch_size = 14 # Must be even, hsa to be split into 2 types of samples
    total_samples = 350
else: # mountain datasets
    size_output_nodes = 3
    samples_per_device = 300 # Amount of samples of each word to send to each device
    total_samples = 360
size_hidden_layer = (650+1)*size_hidden_nodes
hidden_layer = (np.random.normal(size=(size_hidden_layer, )) * np.sqrt(2./650)).astype('float32')

logger.debug("\nCentralized Training: dataset {}, Total Round {}, data_per_round {}, batch size {}". format(experiment, epoch_size * int(samples_per_device/batch_size), batch_size, step_size))

# # We add an extra layer at the server-side model
# neuron_layer_2nd = 2 * size_hidden_nodes

# size_layer_2nd = (size_hidden_nodes+1)*neuron_layer_2nd
# layer_2nd = np.random.uniform(-0.5, 0.5, size_layer_2nd).astype('float32')
# layer_2nd_weight_updates = np.zeros_like(layer_2nd)

#TODO: we will use pytorch on server-side model to automate the training.

# initialize server-side model #TODO: step - 1 [we simply split current architecture] step - 2: after this is done, we extend architecture using pytorch.
size_output_layer = (size_hidden_nodes+1)*size_output_nodes # why we need one more row
output_layer = np.random.uniform(-0.5, 0.5, size_output_layer).astype('float32')
output_weight_updates  = np.zeros_like(output_layer)



logger.debug("Model Setting: model_type: {}, momentum {}, lr {}, number_hidden {}, hidden_size {}". format(model_type, momentum, learningRate, number_hidden, hidden_size))

def init_weights(m):
    if isinstance(m, nn.Linear):
      init.kaiming_normal(m.weight)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv1d):
      init.kaiming_normal(m.weight)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
      init.kaiming_normal(m.weight)
      if m.bias is not None:
        m.bias.data.zero_()
# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.uniform_(m.weight, a = -0.5, b = 0.5)
#         torch.nn.init.uniform_(m.bias, a = -0.5, b = 0.5)
#     if type(m) == nn.Conv2d:
#         torch.nn.init.uniform_(m.weight, a = -0.5, b = 0.5)
#         torch.nn.init.uniform_(m.bias, a = -0.5, b = 0.5)

class client_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self):
        super(client_model, self).__init__()

        model_list = []
        
        model_list.append(nn.Linear(650, size_hidden_nodes, bias = True))
        model_list.append(nn.ReLU())

        self.client = nn.Sequential(*model_list)

    def forward(self, x):
        out = self.client(x)
        return out


class server_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, num_class = 3, number_hidden = 1, hidden_size = 128, input_size = 25):
        super(server_model, self).__init__()

        last_layer_input_size = input_size
        model_list = []

        for _ in range(number_hidden):
            model_list.append(nn.Linear(last_layer_input_size, hidden_size, bias = True))
            model_list.append(nn.ReLU())
            last_layer_input_size = hidden_size
        
        model_list.append(nn.Linear(last_layer_input_size, num_class, bias = True))
        # model_list.append(nn.Sigmoid())

        self.server = nn.Sequential(*model_list)

        logger.debug("server:")
        logger.debug(str(self.server))
    def forward(self, x):
        out = self.server(x)
        return out

class client_conv_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self):
        super(client_conv_model, self).__init__()

        model_list = []
        model_list.append(nn.Conv1d(13, 4, kernel_size = 3, padding="same"))
        model_list.append(nn.ReLU())

        self.client = nn.Sequential(*model_list)

    def forward(self, x):
        # x = x.view(x.size(0), 13, 50)
        out = self.client(x)
        return out

class server_conv_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, num_class = 3, number_hidden = 0, hidden_size = 128, input_size = (50, 13)):
        super(server_conv_model, self).__init__()

        last_layer_input_size = input_size
        model_list = []
        last_layer_input_size = 800
        model_list.append(nn.Conv1d(4, 8, kernel_size = 3, padding="same"))
        model_list.append(nn.BatchNorm1d(8))
        model_list.append(nn.ReLU())
        model_list.append(nn.Conv1d(8, 4, kernel_size = 3, padding="same"))
        model_list.append(nn.BatchNorm1d(4))
        model_list.append(nn.ReLU())
        model_list.append(nn.Flatten(1))
        last_layer_input_size = 200
        for _ in range(number_hidden):
            model_list.append(nn.Linear(last_layer_input_size, hidden_size, bias = True))
            # model_list.append(nn.Dropout(0.5))
            model_list.append(nn.ReLU())
            last_layer_input_size = hidden_size
        
        model_list.append(nn.Linear(last_layer_input_size, num_class, bias = True))

        self.server = nn.Sequential(*model_list)

        print("server:")
        print(self.server)
    def forward(self, x):
        out = self.server(x)
        return out


class client_conv2d_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self):
        super(client_conv2d_model, self).__init__()

        model_list = []
        model_list.append(nn.Conv2d(1, 4, kernel_size = 3, stride = 2, bias = False))
        model_list.append(nn.ReLU())

        self.client = nn.Sequential(*model_list)

    def forward(self, x):
        # x = x.view(x.size(0), 13, 50)
        out = self.client(x)
        return out

class server_conv2d_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, num_class = 3, number_hidden = 0, hidden_size = 128, input_size = (50, 13)):
        super(server_conv2d_model, self).__init__()

        last_layer_input_size = input_size
        model_list = []
        last_layer_input_size = 800
        # model_list.append(nn.Conv2d(4, 8, kernel_size = 3, stride = 1, bias = False))
        # model_list.append(nn.BatchNorm2d(8))
        # model_list.append(nn.ReLU())
        model_list.append(nn.Conv2d(4, 16, kernel_size = 3, stride = 2, bias = False))
        model_list.append(nn.BatchNorm2d(16))
        model_list.append(nn.ReLU())
        model_list.append(nn.Flatten(1))
        last_layer_input_size = 352
        for _ in range(number_hidden):
            model_list.append(nn.Linear(last_layer_input_size, hidden_size, bias = True))
            # model_list.append(nn.Dropout(0.25))
            model_list.append(nn.ReLU())
            last_layer_input_size = hidden_size
        
        model_list.append(nn.Linear(last_layer_input_size, num_class, bias = True))

        self.server = nn.Sequential(*model_list)

        print("server:")
        print(self.server)
    def forward(self, x):
        out = self.server(x)
        return out

if model_type == "fc":
    s_model = server_model(num_class = size_output_nodes, number_hidden = number_hidden, hidden_size = hidden_size, input_size = size_hidden_nodes)

    c_model = client_model()
elif model_type == "conv1d":
    s_model = server_conv_model(num_class = size_output_nodes, number_hidden = number_hidden, hidden_size = hidden_size)

    c_model = client_conv_model()
elif model_type == "conv2d":
    s_model = server_conv2d_model(num_class = size_output_nodes, number_hidden = number_hidden, hidden_size = hidden_size)

    c_model = client_conv2d_model()

c_model.apply(init_weights)
s_model.apply(init_weights)

s_optimizer = torch.optim.SGD(list(s_model.parameters()), lr=learningRate, momentum=momentum, weight_decay=5e-4)
c_optimizer = torch.optim.SGD(list(c_model.parameters()), lr=learningRate, momentum=momentum, weight_decay=5e-4)


pauseListen = False # So there are no threads reading the serial input at the same time
all_files = [file for file in os.listdir("datasets/words/all") ]
must_files = [file for file in os.listdir("datasets/words/must")]
never_files = [file for file in os.listdir("datasets/words/never")]
none_files = [file for file in os.listdir("datasets/words/none")]
only_files = [file for file in os.listdir("datasets/words/only")]
montserrat_files = [file for file in os.listdir("datasets/mountains") if file.startswith("montserrat")]
pedraforca_files = [file for file in os.listdir("datasets/mountains") if file.startswith("pedraforca")]
vermell_files = [file for file in os.listdir("datasets/colors") if file.startswith("vermell")]
verd_files = [file for file in os.listdir("datasets/colors") if file.startswith("verd")]
blau_files = [file for file in os.listdir("datasets/colors") if file.startswith("blau")]
test_montserrat_files = [file for file in os.listdir("datasets/test/") if file.startswith("montserrat")]
test_pedraforca_files = [file for file in os.listdir("datasets/test") if file.startswith("pedraforca")]

# digits_silence_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("silence")]
# digits_one_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("one")]
# digits_two_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("two")]
# digits_three_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("three")]
# digits_four_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("four")]
# digits_five_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("five")]
# digits_unknown_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("unknown")]

# digits_silence_files_EN = [file for file in os.listdir("datasets/CN_digits") if file.startswith("silence") and int(file.split(".")[1])>=500]
# digits_one_files_EN = [file for file in os.listdir("datasets/CN_digits") if file.startswith("one") and int(file.split(".")[1])>=500]
# digits_two_files_EN = [file for file in os.listdir("datasets/CN_digits") if file.startswith("two") and int(file.split(".")[1])>=500]
# digits_three_files_EN = [file for file in os.listdir("datasets/CN_digits") if file.startswith("three") and int(file.split(".")[1])>=500]
# digits_four_files_EN = [file for file in os.listdir("datasets/CN_digits") if file.startswith("four") and int(file.split(".")[1])>=500]
# digits_five_files_EN = [file for file in os.listdir("datasets/CN_digits") if file.startswith("five") and int(file.split(".")[1])>=500]
# digits_unknown_files_EN = [file for file in os.listdir("datasets/CN_digits") if file.startswith("unknown") and int(file.split(".")[1])>=500]

digits_silence_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("silence") and int(file.split(".")[1])>=500]
digits_one_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("one") and int(file.split(".")[1])>=500]
digits_two_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("two") and int(file.split(".")[1])>=500]
digits_three_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("three") and int(file.split(".")[1])>=500]
digits_four_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("four") and int(file.split(".")[1])>=500]
digits_five_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("five") and int(file.split(".")[1])>=500]
digits_unknown_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("unknown") and int(file.split(".")[1])>=500]


# digits_silence_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("silence") and int(file.split(".")[1])<500]
# digits_one_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("one") and int(file.split(".")[1])<500]
# digits_two_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("two") and int(file.split(".")[1])<500]
# digits_three_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("three") and int(file.split(".")[1])<500]
# digits_four_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("four") and int(file.split(".")[1])<500]
# digits_five_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("five") and int(file.split(".")[1])<500]
# digits_unknown_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("unknown") and int(file.split(".")[1])<500]


# digits_silence_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("silence") and int(file.split(".")[1])>=500]
# digits_one_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("one") and int(file.split(".")[1])>=500]
# digits_two_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("two") and int(file.split(".")[1])>=500]
# digits_three_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("three") and int(file.split(".")[1])>=500]
# digits_four_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("four") and int(file.split(".")[1])>=500]
# digits_five_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("five") and int(file.split(".")[1])>=500]
# digits_unknown_files = [file for file in os.listdir("datasets/EN_digits") if file.startswith("unknown") and int(file.split(".")[1])>=500]



random.shuffle(digits_silence_files)
random.shuffle(digits_one_files)
random.shuffle(digits_two_files)
random.shuffle(digits_three_files)
random.shuffle(digits_four_files)
random.shuffle(digits_five_files)
random.shuffle(digits_unknown_files)

graph = []
repaint_graph = True

val_graph = []

random.shuffle(montserrat_files)
random.shuffle(pedraforca_files)
mountains = list(sum(zip(montserrat_files, pedraforca_files), ()))
test_mountains = list(sum(zip(test_montserrat_files, test_pedraforca_files), ()))


def convert_string_to_array(string, one_hot = False):
    global size_output_nodes
    if not one_hot:
        out_act = np.fromstring(string, dtype=float, sep=' ')
        return out_act
    else:
        out_label = np.zeros((size_output_nodes,))
        # string = int.from_bytes(string, "big")
        # print(string)
        out_label[int(string.replace('b', '').replace('\'', '')) - 1] = 1 
        return out_label


def server_compute(Hidden, target, only_forward = False):
    input = torch.tensor(Hidden, requires_grad=True).float()
    label = torch.argmax(torch.from_numpy(target).float())
    s_model.train()
    s_optimizer.zero_grad()
    
    input.retain_grad()
    output = s_model(input)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)

    error = loss.detach().numpy()
    
    if not only_forward:
        if input.grad is not None:
            input.grad.zero_()
        loss.backward(retain_graph = True)

        error_array = input.grad.detach().numpy().astype('float32') # get gradient, the -1 is important, since updates are added to the weights in cpp.

        s_optimizer.step()
        # print("logits:", output.detach().numpy())
        # print("error_array:", error_array)
        accu = torch.argmax(output) == label
        accu = accu.detach().numpy()
        return accu, error, error_array
    else:
        accu = torch.argmax(output) == label
        accu = accu.detach().numpy()
        return accu, error


def server_validate(test_in, test_out):
    # multiple_batch
    input = torch.tensor(test_in).float()
    
    label = torch.from_numpy(test_out).view(input.size(0),).long()

    s_model.eval()

    c_model.eval()
    with torch.no_grad():
        output = s_model(c_model(input))
        
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, label)

    error = loss.detach().numpy() / input.size(0)


    accu = (torch.argmax(output, dim = 1) == label).sum() / input.size(0)

    accu = accu.detach().numpy()

    return error, accu

def server_train(train_in, train_out):

    input = torch.tensor(train_in).float()
    
    label = torch.from_numpy(train_out).view(input.size(0),).long()

    s_model.train()
    c_model.train()

    s_optimizer.zero_grad()
    c_optimizer.zero_grad()

    output = s_model(c_model(input))
    
    criterion = nn.CrossEntropyLoss()

    loss = criterion(output, label)

    error = loss.detach().numpy()

    loss.backward()

    s_optimizer.step()
    c_optimizer.step()
    
    accu = (torch.argmax(output, dim = 1) == label).sum()
    accu = accu.detach().numpy()
    # print(accu)
    return error, accu
    
def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        # logger.debug(msg)
        if msg[:-2] == keyword:
            break
        else:
            print(f'({arduino.port}):',msg, end='')

def init_network(hidden_layer, output_layer, device, deviceIndex):
    device.reset_input_buffer()
    time.sleep(1.0)
    device.write(b's')
    time.sleep(0.1)
    # 
    print_until_keyword('start', device)
    # modelReceivedConfirmation = device.readline().decode()
    # print(f"Model received confirmation: ", modelReceivedConfirmation)
    # print(f"Sending model to {device.port}")

    device.write(struct.pack('f', learningRate))
    device.write(struct.pack('f', momentum))
    device.read() # wait until confirmation of float received

    for i in range(len(hidden_layer)):
        device.read() # wait until confirmation of float received
        float_num = hidden_layer[i]
        data = struct.pack('f', float_num)
        device.write(data)

    # print(f"Client-side Model sent to {device.port}")
    modelReceivedConfirmation = device.readline().decode()
    # print(f"Model received confirmation: ", modelReceivedConfirmation)


def sendSamplesIIDCustom(device, deviceIndex, batch_size, batch_index):
    global all_files, must_files, never_files, none_files, only_files

    # each_sample_amt = int(batch_size/2)

    start = (deviceIndex*samples_per_device) + (batch_index * batch_size)
    end = (deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size
    real_start = start // 5
    real_end = (end - start) // 5 + start // 5
    # print(f"[{device.port}] Sending samples from {start} to {end}")
    for i in range(real_start, real_end):
        filename = all_files[i]
        num_button = 1
        sendSample(device, 'datasets/words/all/'+filename, num_button, deviceIndex)

        filename = must_files[i]
        num_button = 2
        sendSample(device, 'datasets/words/must/'+filename, num_button, deviceIndex)

        filename = never_files[i]
        num_button = 3

        sendSample(device, 'datasets/words/never/'+filename, num_button, deviceIndex)

        filename = none_files[i]
        num_button = 4
        sendSample(device, 'datasets/words/none/'+filename, num_button, deviceIndex)

        filename = only_files[i]
        num_button = 5
        sendSample(device, 'datasets/words/only/'+filename, num_button, deviceIndex)


def sendSamplesIIDDigits(device, deviceIndex, batch_size, batch_index):
    global digits_silence_files, digits_one_files, digits_two_files, digits_three_files, digits_four_files, digits_five_files, digits_unknown_files

    # each_sample_amt = int(batch_size/2)

    start = (deviceIndex*samples_per_device) + (batch_index * batch_size)
    end = (deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size
    real_start = start // 7
    real_end = (end - start) // 7 + start // 7
    for i in range(real_start, real_end):
        filename = digits_silence_files[i]
        num_button = 1
        sendSample(device, 'datasets/CN_digits/'+filename, num_button, deviceIndex)
        
        filename = digits_one_files[i]
        num_button = 2
        sendSample(device, 'datasets/CN_digits/'+filename, num_button, deviceIndex)

        filename = digits_two_files[i]
        num_button = 3
        sendSample(device, 'datasets/CN_digits/'+filename, num_button, deviceIndex)

        filename = digits_three_files[i]
        num_button = 4
        sendSample(device, 'datasets/CN_digits/'+filename, num_button, deviceIndex)

        filename = digits_four_files[i]
        num_button = 5
        sendSample(device, 'datasets/CN_digits/'+filename, num_button, deviceIndex)

        filename = digits_five_files[i]
        num_button = 6
        sendSample(device, 'datasets/CN_digits/'+filename, num_button, deviceIndex)

        filename = digits_unknown_files[i]
        num_button = 7
        sendSample(device, 'datasets/CN_digits/'+filename, num_button, deviceIndex)

# Batch size: The amount of samples to send
def sendSamplesIID(device, deviceIndex, batch_size, batch_index):
    global montserrat_files, pedraforca_files, mountains

    # each_sample_amt = int(batch_size/2)

    start = (deviceIndex*samples_per_device) + (batch_index * batch_size)
    end = (deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size

    files = mountains[start:end]
    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        else:
            exit("Unknown button for sample")
        sendSample(device, 'datasets/mountains/'+filename, num_button, deviceIndex)

def getSamplesIID(batch_size, batch_start_index):
    global montserrat_files, pedraforca_files, mountains

    # each_sample_amt = int(batch_size/2)

    start = batch_start_index
    end = batch_start_index + batch_size
    
    input_list = []
    label_list = []

    files = mountains[start:end]
    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        else:
            exit("Unknown button for sample")
        
        if num_button == 1:
            input_array = np.load("processed_datasets/mountains/montserrat_{}.npy".format(filename.split("/")[-1].split(".")[1]))
        elif num_button == 2:
            input_array = np.load("processed_datasets/mountains/pedraforca_{}.npy".format(filename.split("/")[-1].split(".")[1]))
        else:
            exit("Unknown button for sample")
    
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        input_list_array = np.concatenate(input_list, axis = 0)
        label_list_array = np.array(label_list).reshape(-1, 1)
    return input_list_array, label_list_array


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
        input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        
        #
        # input_array2 = MFCC(filename)
        # input_array2 == input_array
        #

        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.
        
        filename = digits_one_files[i]
        num_button = 2
        input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_two_files[i]
        num_button = 3
        input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_three_files[i]
        num_button = 4
        input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_four_files[i]
        num_button = 5
        input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_five_files[i]
        num_button = 6
        input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

        filename = digits_unknown_files[i]
        num_button = 7
        input_array = np.load("processed_datasets/{}/{}.npy".format(experiment,filename.split("/")[-1].replace(".json", "")))
        input_list.append(input_array)
        label_list.append(num_button - 1) # need to minus oen to act as label.

    input_list_array = np.concatenate(input_list, axis = 0)
    label_list_array = np.array(label_list).reshape(-1, 1)
    return input_list_array, label_list_array


def sendSamplesNonIID(device, deviceIndex, batch_size, batch_index):
    global montserrat_files, pedraforca_files, vermell_files, verd_files, blau_files

    start = (batch_index * batch_size)
    end = (batch_index * batch_size) + batch_size

    dir = 'datasets/' # TMP fix
    if (deviceIndex == 0):
        files = vermell_files[start:end]
        num_button = 1
        dir = 'colors'
    elif  (deviceIndex == 1):
        files = montserrat_files[start:end]
        num_button = 2
        dir = 'mountains'
    elif  (deviceIndex == 2):
        files = pedraforca_files[start:end]
        num_button = 3
        dir = 'mountains'
    else:
        exit("Exceeded device index")

    for i, filename in enumerate(files):
        sendSample(device, f"datasets/{dir}/{filename}", num_button, deviceIndex)

def sendSample(device, samplePath, num_button, deviceIndex, only_forward = False):
    global running_batch_accu
    with open(samplePath) as f:
        data = json.load(f)
        device.write(b't')
        startConfirmation = device.readline().decode()

        device.write(struct.pack('B', num_button))
        button_confirm = device.readline().decode()

        device.write(struct.pack('B', 1 if only_forward else 0))
        only_forward_confirm = device.readline().decode()

        if 'payload' in data:
            for i, value in enumerate(data['payload']['values']):
                device.write(struct.pack('h', value))
        else:
            for i, value in enumerate(data['values']):
                device.write(struct.pack('h', value))
        sample_received_confirm = device.readline().decode()
        

        #Receive input from client (uncomment corresponding part (line 141) in main.ino)
        # input_list = []
        # for _ in range(50):
        #     inputs = device.readline().decode()
        #     inputs_converted = convert_string_to_array(inputs)
        #     input_list.append(inputs_converted)
        # input_array = np.concatenate(input_list, axis = 0).reshape(1,650)

        # if not os.path.isdir("processed_datasets/{}".format(experiment)):
        #     os.makedirs("processed_datasets/{}".format(experiment))
        # np.save("processed_datasets/{}/{}.npy".format(experiment, samplePath.split("/")[-1].replace(".json", "")), input_array)

        # Receive activation from client
        outputs = device.readline().decode()

        if only_forward:
            # Perform server-side computation (forward)
            hidden_activation = convert_string_to_array(outputs)
            label = convert_string_to_array(str(num_button), one_hot = True)
            forward_accu, forward_error = server_compute(hidden_activation, label, only_forward= True)
        else:
            # Receive label from client
            nb = device.readline()[:-2]

            # Perform server-side computation (forward/backward)
            hidden_activation = convert_string_to_array(outputs)
            label = convert_string_to_array(str(nb), one_hot = True)
            forward_accu, forward_error, error_array = server_compute(hidden_activation, label, only_forward= False)
            
            # Send Error Array to client to continue backward #TODO: implement this
            for i in range(size_hidden_nodes): # hidden layer
                d.read() # wait until confirmatio
                float_num = error_array[i]
                data = struct.pack('f', float_num)
                d.write(data)

        device.readline().decode() # Accept 'Done' command

        ne = device.readline()[:-2]

        n_epooch = int(ne)
        running_batch_accu += forward_accu
        graph.append([n_epooch, forward_error, deviceIndex])

def sendTestSamples(device, deviceIndex):
    global test_mountains

    start = deviceIndex*40
    end = (deviceIndex*40) + 40

    files = mountains[start:end]
    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        else:
            exit("Unknown button for sample")
        sendSample(device, 'datasets/mountains/'+filename, num_button, deviceIndex, True)

def read_number(msg):
    while True:
        try:
            #return 2;
            return int(input(msg))
        except:
            print("ERROR: Not a number")

def read_port(msg):
    while True:
        try:
            port = input(msg)
            #port = "COM3";
            return serial.Serial(port, 9600)
        except:
            print(f"ERROR: Wrong port connection ({port})")

def plot_graph():
    global graph, repaint_graph, devices

    if (repaint_graph):
        colors = ['r', 'g', 'b', 'y']
        markers = ['-', '--', ':', '-.']
        #devices =  [x[2] for x in graph]        
        for device_index, device in enumerate(devices):
            epoch = [x[0] for x in graph if x[2] == device_index]
            error = [x[1] for x in graph if x[2] == device_index]
        
            plt.plot(error, colors[device_index] + markers[device_index], label=f"Device {device_index}")

        plt.legend()
        plt.xlim(left=0)
        plt.ylim(bottom=0, top=2.0)
        plt.ylabel('Loss') # or Error
        plt.xlabel('Image')
        # plt.axes().set_ylim([0, 0.6])
        # plt.xlim(bottom=0)
        # plt.autoscale()
        repaint_graph = False

    plt.pause(2)

def plot_train_accu():
    global running_batch_accu_list

    # if (repaint_graph):
    colors = ['r', 'g', 'b', 'y']
    markers = ['-', '--', ':', '-.']
    #devices =  [x[2] for x in graph]
    # running_batch_accu_list
    # for device_index, device in enumerate(devices):
    #     epoch = [x[0] for x in graph if x[2] == device_index]
    #     error = [x[1] for x in graph if x[2] == device_index]
    
    plt.plot(running_batch_accu_list, 'g-', label=f"Train_Accu")

    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1.0)
    plt.ylabel('Accu') # or Error
    plt.xlabel('Image')
    # plt.axes().set_ylim([0, 0.6])
    # plt.xlim(bottom=0)
    # plt.autoscale()
    # repaint_graph = False

    plt.pause(2)

def plot_val_graph():
    global val_graph, repaint_graph

    colors = ['r', 'g', 'b', 'y']
    markers = ['-', '--', ':', '-.']
    error = [x[0] for x in val_graph]
    accuracy = [x[1] for x in val_graph]
    
    plt.plot(accuracy, colors[1] + markers[0], label="accuracy")
    plt.plot(error, colors[2] + markers[0], label="error")
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1.0)
    plt.ylabel('Accuracy') # or Error
    plt.xlabel('Epoch')

    plt.pause(2)

def listenDevice(device, deviceIndex):
    global pauseListen, graph
    while True:
        while (pauseListen):
            print("Paused...")
            time.sleep(0.1)

        d.timeout = None
        msg = device.readline().decode()
        if (len(msg) > 0):
            print(f'({device.port}):', msg, end="")
            # Modified to graph
            # if msg[:-2] == 'graph':
            #     read_graph(device, deviceIndex)

            # el
            if msg[:-2] == 'start_fl':
                startFL()

def getDevices():
    global devices, devices_connected
    num_devices = read_number("Number of devices: ")

    available_ports = comports()
    print("Available ports:")
    for available_port in available_ports:
        print(available_port)

    devices = [read_port(f"Port device_{i+1}: ") for i in range(num_devices)]
    devices_connected = devices

def FlGetModel(d, device_index, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected):
    global size_hidden_layer, size_output_layer
    # d.reset_input_buffer()
    # d.reset_output_buffer()
    d.timeout = 5

    # print(f'Starting connection to {d.port} ...') # Hanshake
    d.write(b'a') # Python --> SYN --> Arduino
    
    if d.read() == b'y': # Python <-- SYN ACK <-- Arduino

        d.write(b's') # Python --> ACK --> Arduino
        
        devices_connected.append(d)
        
        d.timeout = None

        print_until_keyword('start', d)
        devices_num_epochs.append(int(d.readline()[:-2]))

        ini_time = time.time()

        for i in range(size_hidden_layer): # hidden layer
            data = d.read(4)
            [float_num] = struct.unpack('f', data)
            devices_hidden_layer[device_index][i] = float_num

        # if it was not connected before, we dont use the devices' model
        if not d in old_devices_connected:
            devices_num_epochs[device_index] = 0
            print(f'Model not used. The device {d.port} has an outdated model')

    else:
        print(f'Connection timed out. Skipping {d.port}.')

def sendModel(d, hidden_layer, output_layer):
    for i in range(size_hidden_layer): # hidden layer
        d.read() # wait until confirmatio
        float_num = hidden_layer[i]
        data = struct.pack('f', float_num)
        d.write(data)


def startFL():
    global devices_connected, hidden_layer, output_layer, pauseListen

    pauseListen = True

    print('Model Aggregation...')
    old_devices_connected = devices_connected
    devices_connected = []
    devices_hidden_layer = np.empty((len(devices), size_hidden_layer), dtype='float32')
    devices_output_layer = np.empty((len(devices), size_output_layer), dtype='float32')
    devices_num_epochs = []
    
    ##################
    # Receiving models
    ##################
    threads = []
    for i, d in enumerate(devices):
        thread = threading.Thread(target=FlGetModel, args=(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end

    
    ####################
    # Processing models
    ####################

    # if sum == 0, any device made any epoch
    if sum(devices_num_epochs) > 0:
        # We can use weights to change the importance of each device
        # example weights = [1, 0.5] -> giving more importance to the first device...
        # is like percentage of importance :  sum(a * weights) / sum(weights)
        ini_time = time.time() * 1000
        hidden_layer = np.average(devices_hidden_layer, axis=0, weights=devices_num_epochs)
        output_layer = np.average(devices_output_layer, axis=0, weights=devices_num_epochs)

    # Doing validation
    c_model.load_state_dict({'client.0.weight': torch.tensor(hidden_layer[:size_hidden_nodes*650]).view(650, size_hidden_nodes).t().float(), 'client.0.bias': torch.tensor(hidden_layer[size_hidden_nodes*650:]).float()})
    
    if experiment == "digits":
        test_in, test_out = getSamplesIIDDigits(70, 630)

        error, accu = server_validate(test_in, test_out)
        logger.debug(f"Validation Accuracy {100 * accu}%\n")
        val_graph.append([error, accu, 0])
    elif experiment == "iid":
        test_in, test_out = getSamplesIID(60, 300)

        error, accu = server_validate(test_in, test_out)
        logger.debug(f"Validation Accuracy {100 * accu}%\n")
        val_graph.append([error, accu, 0])

    #################
    # Sending models
    #################
    threads = []
    for d in devices_connected:
        thread = threading.Thread(target=sendModel, args=(d, hidden_layer, output_layer))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end


    pauseListen = False

def set_button(device, button):
    device.write(b'b')
    print_until_keyword('bt_set', device)
    device.write(button)
    print(f"press/hold botton {button} on {device.port}")


devices = [0]

# Train the device
if experiment == 'digits':
    train_in, train_out = getSamplesIIDDigits(samples_per_device, 0)
    test_in, test_out = getSamplesIIDDigits(total_samples - samples_per_device, samples_per_device)

if model_type == "conv1d":
    train_in = np.reshape(train_in, (samples_per_device, 13, 50))
    # train_in = np.reshape(train_in, (samples_per_device, 50, 13)).transpose(0, 2, 1)
    test_in = np.reshape(test_in, (total_samples - samples_per_device, 13, 50))
    # test_in = np.reshape(test_in, (total_samples - samples_per_device, 50, 13)).transpose(0, 2, 1)
elif model_type == "conv2d":
    train_in = np.reshape(train_in, (samples_per_device, 1, 13, 50))
    # train_in = np.reshape(train_in, (samples_per_device, 50, 13)).transpose(0, 2, 1)
    test_in = np.reshape(test_in, (total_samples - samples_per_device, 1, 13, 50))
    # test_in = np.reshape(test_in, (total_samples - samples_per_device, 50, 13)).transpose(0, 2, 1)



init_time = time.time()

for epoch in range(epoch_size):
    
    total_round = int(samples_per_device/batch_size)
    
    # shuffle train data
    permute_idx = np.random.permutation(samples_per_device)
    train_in[:, ] = train_in[permute_idx, ]
    train_out[:, ] = train_out[permute_idx, ]

    for batch in range(total_round):
        logger.debug("Epoch {}/{}, Round {}/{} (data per round: {})".format(epoch, epoch_size, batch, total_round, batch_size))
        running_batch_accu = 0
        
        for step in range(batch_size//step_size):
            batch_train_in = train_in[batch*batch_size+step*step_size:batch*batch_size+(step+1)*step_size,:]
            batch_train_out = train_out[batch*batch_size+step*step_size:batch*batch_size+(step+1)*step_size,:]

            train_error, train_accu = server_train(batch_train_in, batch_train_out)

            running_batch_accu += train_accu
            
            graph.append([batch, train_error, 0])

        val_error, val_accu = server_validate(test_in, test_out)

        logger.debug(f"Validation Accuracy {100 * val_accu}%\n")

        val_graph.append([val_error, val_accu, 0])
    
        logger.debug("Training Accuracy is {}%".format(100 * running_batch_accu/batch_size))
        # print(running_batch_accu, batch_size)
        running_batch_accu_list.append(running_batch_accu/batch_size)
        
train_time = time.time() - init_time

plt.figure(1)
plt.ion()
plt.show()

font_sm = 13
font_md = 16
font_xl = 18
plt.rc('font', size=font_sm)          # controls default text sizes
plt.rc('axes', titlesize=font_sm)     # fontsize of the axes title
plt.rc('axes', labelsize=font_md)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_sm)    # legend fontsize
plt.rc('figure', titlesize=font_xl)   # fontsize of the figure title

plot_graph()
figname = f"newplots/ES{epoch_size}-BS{batch_size}-LR{learningRate}-M{momentum}-NH{number_hidden}-HS{hidden_size}-HN{size_hidden_nodes}-TT{train_time}-{experiment}_train.eps"
plt.savefig(figname, format='eps')
logger.debug(f"Generated {figname}")

plt.figure(2)
plt.ion()
plt.show()
plt.rc('font', size=font_sm)          # controls default text sizes
plt.rc('axes', titlesize=font_sm)     # fontsize of the axes title
plt.rc('axes', labelsize=font_md)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_sm)    # legend fontsize
plt.rc('figure', titlesize=font_xl)   # fontsize of the figure title
plot_val_graph()
figname2 = f"newplots/ES{epoch_size}-BS{batch_size}-LR{learningRate}-M{momentum}-NH{number_hidden}-HS{hidden_size}-HN{size_hidden_nodes}-TT{train_time}-{experiment}_val.eps"
plt.savefig(figname2, format='eps')
print(f"Generated {figname2}")

plt.figure(3)
plt.ion()
plt.show()
plt.rc('font', size=font_sm)          # controls default text sizes
plt.rc('axes', titlesize=font_sm)     # fontsize of the axes title
plt.rc('axes', labelsize=font_md)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_sm)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_sm)    # legend fontsize
plt.rc('figure', titlesize=font_xl)   # fontsize of the figure title
plot_train_accu()
figname3 = f"newplots/ES{epoch_size}-BS{batch_size}-LR{learningRate}-M{momentum}-NH{number_hidden}-HS{hidden_size}-HN{size_hidden_nodes}-TT{train_time}-{experiment}_train_accu.eps"
plt.savefig(figname3, format='eps')
logger.debug(f"Generated {figname3}")


