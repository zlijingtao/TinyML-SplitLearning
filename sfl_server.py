import serial
from serial.tools.list_ports import comports

import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
import sys
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


random.seed(4321)
np.random.seed(4321)

epoch_size = 3 # default = 1

batch_size = 10 # Must be even, hsa to be split into 2 types of samples
running_batch_accu = 0
running_batch_accu_list = []
experiment = 'digits' # 'iid', 'no-iid', 'train-test', 'custom', 'digits'

# initialize client-side model
size_hidden_nodes = 25
if experiment == "custom":
    size_output_nodes = 5
    samples_per_device = 250 # Amount of samples of each word to send to each device
elif experiment == 'digits':
    samples_per_device = 630 # Amount of samples of each word to send to each device
    size_output_nodes = 7
    batch_size = 14 # Must be even, hsa to be split into 2 types of samples
else: # mountain datasets
    size_output_nodes = 3
    samples_per_device = 300 # Amount of samples of each word to send to each device

size_hidden_layer = (650+1)*size_hidden_nodes
hidden_layer = (np.random.normal(size=(size_hidden_layer, )) * np.sqrt(2./650)).astype('float32')

logger.debug("\nTraining Setting: dataset {}, Total Round {}, data_per_round {}". format(experiment, epoch_size * int(samples_per_device/batch_size), batch_size))

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

momentum = 0.7
learningRate= 0.01
number_hidden = 0
hidden_size = 64

logger.debug("Model Setting: momentum {}, lr {}, number_hidden {}, hidden_size {}". format(momentum, learningRate, number_hidden, hidden_size))

def init_weights(m):
    if isinstance(m, nn.Linear):
      init.kaiming_normal(m.weight)
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


s_model = server_model(num_class = size_output_nodes, number_hidden = number_hidden, hidden_size = hidden_size, input_size = size_hidden_nodes)
s_model.apply(init_weights)
c_model = client_model()
# print(c_model.state_dict())
c_model.load_state_dict({'client.0.weight': torch.tensor(hidden_layer[:size_hidden_nodes*650]).view(650, size_hidden_nodes).t().float(), 'client.0.bias': torch.tensor(hidden_layer[size_hidden_nodes*650:]).float()})
s_optimizer = torch.optim.SGD(list(s_model.parameters()), lr=learningRate, momentum=momentum, weight_decay=5e-4)
# s_optimizer = torch.optim.Adam(list(s_model.parameters()), lr=learningRate)


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

digits_silence_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("silence")]
digits_one_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("one")]
digits_two_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("two")]
digits_three_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("three")]
digits_four_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("four")]
digits_five_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("five")]
digits_unknown_files = [file for file in os.listdir("datasets/CN_digits") if file.startswith("unknown")]

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

    output = s_model(c_model(input))
    
    criterion = nn.CrossEntropyLoss()

    loss = criterion(output, label)

    error = loss.detach().numpy() / input.size(0)

    accu = (torch.argmax(output, dim = 1) == label).sum() / input.size(0)

    accu = accu.detach().numpy()

    return error, accu

def server_compute_old(Hidden, target, only_forward = False):

    #TODO: matrix multiplication optimization

    # Compute forward and error
    error = 0.0
    output_array = np.zeros((size_output_nodes,))
    output_delta_array = np.zeros((size_output_nodes,))
    for i in range(size_output_nodes):
        # Compute bias
        accu = output_layer[size_hidden_nodes * size_output_nodes + i]
        for j in range(size_hidden_nodes):
            accu += Hidden[j] * output_layer[j*size_output_nodes + i] #[1, 25] * [25, 3]

        output_array[i] = 1.0 / (1.0 + np.exp(-accu))
        output_delta_array[i] = (target[i] - output_array[i]) * output_array[i] * (1.0 - output_array[i])
        error += 1/size_output_nodes * (target[i] - output_array[i]) * (target[i] - output_array[i])

    
    if not only_forward:
        # Compute backward and gradients w.r.t. to activaiton (error_array) to client
        error_array = np.zeros((size_hidden_nodes,))
        for i in range(size_hidden_nodes):
            for j in range(size_output_nodes):
                error_array[i] += output_layer[i*size_output_nodes + j] * output_delta_array[j] #[25, 3] * [3, 1]
        
        # Update weights
        for i in range(size_output_nodes):
            output_weight_updates[size_hidden_nodes * size_output_nodes + i] = learningRate * output_delta_array[i] + momentum * output_weight_updates[size_hidden_nodes * size_output_nodes + i] #bias update
            output_layer[size_hidden_nodes * size_output_nodes + i] += output_weight_updates[size_hidden_nodes * size_output_nodes + i]
            for j in range(size_hidden_nodes):
                output_weight_updates[j*size_output_nodes + i] = learningRate * Hidden[j] * output_delta_array[i] + momentum * output_weight_updates[j*size_output_nodes + i]
                output_layer[j*size_output_nodes + i] = output_weight_updates[j*size_output_nodes + i]

    if not only_forward:
        return error, error_array
    else:
        return error

    

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
    running_batch_accu_list
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

# getDevices()

global devices, devices_connected
num_devices = 1

available_ports = comports()
print("Available ports:")
for available_port in available_ports:
    print(available_port)

try:
    print("Access default port")
    devices = [serial.Serial('/dev/ttyACM0', 9600)]
except:
    print("Access alternative port")
    devices = [serial.Serial('/dev/ttyACM1', 9600)]
devices_connected = devices




# Send the blank model to all the devices
threads = []
print("Number of devices is {}".format(str(devices)))
for i, d in enumerate(devices):
    thread = threading.Thread(target=init_network, args=(hidden_layer, output_layer, d, i))
    thread.daemon = True
    thread.start()
    threads.append(thread)
for thread in threads: thread.join() # Wait for all the threads to end

ini_time = time.time()

# # Press a dummy virtual button to start the loop
# threads = []
# for deviceIndex, device in enumerate(devices):
#     thread = threading.Thread(target=set_button, args=(device, 5))
#     thread.daemon = True
#     thread.start()
#     threads.append(thread)
# for thread in threads: thread.join() # Wait for all the threads to end


# Train the device
for epoch in range(epoch_size):
    total_round = int(samples_per_device/batch_size)
    for batch in range(total_round):
        running_batch_accu = 0
        logger.debug("Epoch {}/{}, Round {}/{} (data per round: {})".format(epoch, epoch_size, batch, total_round, batch_size))
        for deviceIndex, device in enumerate(devices):
            if experiment == 'iid' or experiment == 'train-test':
                thread = threading.Thread(target=sendSamplesIID, args=(device, deviceIndex, batch_size, batch))
            elif experiment == 'no-iid':
                thread = threading.Thread(target=sendSamplesNonIID, args=(device, deviceIndex, batch_size, batch))
            elif experiment == 'custom':
                thread = threading.Thread(target=sendSamplesIIDCustom, args=(device, deviceIndex, batch_size, batch))
            elif experiment == 'digits':
                thread = threading.Thread(target=sendSamplesIIDDigits, args=(device, deviceIndex, batch_size, batch))

            thread.daemon = True
            thread.start()
            threads.append(thread)
        for thread in threads: thread.join() # Wait for all the threads to end
        
        logger.debug("Training Accuracy is {}%".format(100 * running_batch_accu/batch_size))
        
        startFL()
        
        running_batch_accu_list.append(running_batch_accu/batch_size)
    
train_time = time.time()-ini_time

if experiment == 'train-test':
    for deviceIndex, device in enumerate(devices):
        thread = threading.Thread(target=sendTestSamples, args=(device, deviceIndex))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end


# Listen their updates
for i, d in enumerate(devices):
    thread = threading.Thread(target=listenDevice, args=(d, i))
    thread.daemon = True
    thread.start()

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


