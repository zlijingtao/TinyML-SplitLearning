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
#TODO: compare baseline validation and more layers (or larger model)
#TODO: try to extinct resources on board. (multi-iamge batch?)
#TODO: find a way to store some image and pull out them to train.
#TODO: Use our own image. 
#TODO: How to make it really useful? Just as the original one, need some button to do the labeling.

random.seed(4321)
np.random.seed(4321)

samples_per_device = 200 # Amount of samples of each word to send to each device
batch_size = 10 # Must be even, hsa to be split into 2 types of samples
running_batch_accu = 0
running_batch_accu_list = []
# experiment = 'iid' # 'iid', 'no-iid', 'train-test'
experiment = 'custom' # 'iid', 'no-iid', 'train-test', 'custom'
sgd_counter = 0
eq_batch_size = 5

# initialize client-side model
size_hidden_nodes = 25
size_output_nodes = 5
size_hidden_layer = (650+1)*size_hidden_nodes
hidden_layer = np.random.uniform(-0.5,0.5, size_hidden_layer).astype('float32')


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

momentum = 0.9
learningRate= 0.02
number_hidden = 2
hidden_size = 128

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
        # model_list.append(nn.ReLU())

        self.client = nn.Sequential(*model_list)

    def forward(self, x):
        out = self.client(x)
        return out


class server_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, num_class = 3, number_hidden = 1, hidden_size = 128, input_size = (50, 13)):
        super(server_model, self).__init__()

        last_layer_input_size = input_size
        model_list = []
        model_list.append(nn.Conv1d(13, 32, kernel_size = 5, padding="same"))
        model_list.append(nn.ReLU())
        # model_list.append(nn.Dropout(0.25))
        last_layer_input_size = 1600
        model_list.append(nn.Conv1d(32, 16, kernel_size = 3, padding="same"))
        model_list.append(nn.ReLU())
        # model_list.append(nn.Dropout(0.25))
        model_list.append(nn.Flatten(0))
        last_layer_input_size = 800
        for _ in range(number_hidden):
            model_list.append(nn.Linear(last_layer_input_size, hidden_size, bias = True))
            model_list.append(nn.ReLU())
            last_layer_input_size = hidden_size
        
        model_list.append(nn.Linear(last_layer_input_size, num_class, bias = True))
        # model_list.append(nn.Sigmoid())

        self.server = nn.Sequential(*model_list)

        print("server:")
        print(self.server)
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

graph = []
repaint_graph = True

val_graph = []

random.shuffle(montserrat_files)
random.shuffle(pedraforca_files)
mountains = list(sum(zip(montserrat_files, pedraforca_files), ()))
test_mountains = list(sum(zip(test_montserrat_files, test_pedraforca_files), ()))
# random.shuffle(vermell_files)
# random.shuffle(verd_files)
# random.shuffle(blau_files)


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


def server_compute(model, optimizer, input, target, only_forward = False):
    global sgd_counter, eq_batch_size
    # global s_model, s_optimizer
    # input = torch.from_numpy(Hidden).cuda()
    input = torch.tensor(input, requires_grad=True).float().cuda()
    
    # label = torch.from_numpy(target).float().cuda()
    label = torch.argmax(torch.from_numpy(target).float()).cuda()

    model.cuda()

    model.train()

    if sgd_counter  == 0:
        optimizer.zero_grad() #TODO: test not perform update but accumualting grads before each updating.
        # use multiple batch = 1 to simulate a batch size > 1 update.

    # input.retain_grad()

    output = model(input)
    
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    loss = criterion(output, label)
    # loss = criterion(output, label) * 1 / size_output_nodes

    error = loss.detach().cpu().numpy()
    
    accu = torch.argmax(output) == label

    accu = accu.detach().cpu().numpy()

    print("output: {}".format(str(output.detach().cpu().numpy())))

    if not only_forward:
        
        loss.backward()
        sgd_counter += 1
        # get gradient
        # error_array = input.grad.detach().cpu().numpy()
        if sgd_counter % eq_batch_size == 0:
            sgd_counter = 0
            optimizer.step()

        

    return accu, error


def server_validate(test_in, test_out):
    global s_model, c_model, s_optimizer
    # multiple_batch
    input = torch.tensor(test_in, requires_grad=True).float().cuda()
    # input = input/100.
    # print(input)
    
    label = torch.from_numpy(test_out).view(input.size(0), ).long().cuda()

    c_model.cuda()

    s_model.cuda()

    s_model.eval()

    c_model.eval()

    output = s_model(c_model(input))
    
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    loss = criterion(output, label)
    # loss = criterion(output, label) * 1 / size_output_nodes

    error = loss.detach().cpu().numpy() / input.size(0)

    accu = (torch.argmax(output, dim = 1) == label).sum() / input.size(0)

    print(torch.argmax(output, dim = 1))
    print(label)

    accu = accu.detach().cpu().numpy()

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
        print(msg)
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
    print(f"Sending model to {device.port}")

    device.write(struct.pack('f', learningRate))
    device.write(struct.pack('f', momentum))
    device.read() # wait until confirmation of float received

    for i in range(len(hidden_layer)):
        device.read() # wait until confirmation of float received
        float_num = hidden_layer[i]
        data = struct.pack('f', float_num)
        device.write(data)

    print(f"Client-side Model sent to {device.port}")
    modelReceivedConfirmation = device.readline().decode()
    print(f"Model received confirmation: ", modelReceivedConfirmation)


def sendSamplesIIDCustom(device, deviceIndex, batch_size, batch_index):
    global all_files, must_files, never_files, none_files, only_files

    # each_sample_amt = int(batch_size/2)

    start = (deviceIndex*samples_per_device) + (batch_index * batch_size)
    end = (deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size
    real_start = start // 5
    real_end = (end - start) // 5 + start // 5
    print(f"[{device.port}] Sending samples from {start} to {end}")
    for i in range(real_start, real_end):
        filename = all_files[i]
        num_button = 1
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(all_files)}): Class 1: all")
        sendSample(device, 'datasets/words/all/'+filename, num_button, deviceIndex)

        filename = must_files[i]
        num_button = 2
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(must_files)}): Class 2: must")
        sendSample(device, 'datasets/words/must/'+filename, num_button, deviceIndex)

        filename = never_files[i]
        num_button = 3
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(never_files)}): Class 3: never")
        sendSample(device, 'datasets/words/never/'+filename, num_button, deviceIndex)

        filename = none_files[i]
        num_button = 4
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(none_files)}): Class 4: none")
        sendSample(device, 'datasets/words/none/'+filename, num_button, deviceIndex)

        filename = only_files[i]
        num_button = 5
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(only_files)}): Class 5: only")
        sendSample(device, 'datasets/words/only/'+filename, num_button, deviceIndex)

# Batch size: The amount of samples to send
def sendSamplesIID(device, deviceIndex, batch_size, batch_index):
    global montserrat_files, pedraforca_files, mountains

    # each_sample_amt = int(batch_size/2)

    start = (deviceIndex*samples_per_device) + (batch_index * batch_size)
    end = (deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size

    print(f"[{device.port}] Sending samples from {start} to {end}")

    files = mountains[start:end]
    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        else:
            exit("Unknown button for sample")
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
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
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        sendSample(device, f"datasets/{dir}/{filename}", num_button, deviceIndex)

def sendSample(device, samplePath, num_button, deviceIndex, only_forward = False):
    global running_batch_accu, s_model, s_optimizer
    with open(samplePath) as f:
        data = json.load(f)
        device.write(b't')
        startConfirmation = device.readline().decode()
        print(f"[{device.port}] Train start confirmation:", startConfirmation)

        device.write(struct.pack('B', num_button))
        print(device.readline().decode()) # Button confirmation

        device.write(struct.pack('B', 1 if only_forward else 0))
        print(f"Only forward confirmation: {device.readline().decode()}") # Button confirmation
        
        if 'payload' in data:
            for i, value in enumerate(data['payload']['values']):
                device.write(struct.pack('h', value))
        else:
            for i, value in enumerate(data['values']):
                device.write(struct.pack('h', value))
        print(f"[{device.port}] Sample received confirmation:", device.readline().decode())
        

        #Receive input from client
        input_list = []
        for _ in range(50):
            inputs = device.readline().decode()
            inputs_converted = convert_string_to_array(inputs)
            input_list.append(inputs_converted)
        input_array = np.transpose(np.concatenate(input_list, axis = 0).reshape(50, 13))
        # input_array = np.concatenate(input_list, axis = 0).reshape(13, 50)
        # .reshape(1,650)
        # print(f"test_input: ", input_array)

        # if num_button == 1:
        #     np.save("processed_datasets/mountains/montserrat_{}".format(samplePath.split("/")[-1].split(".")[1]), input_array)
        # elif num_button == 2:
        #     np.save("processed_datasets/mountains/pedraforca_{}".format(samplePath.split("/")[-1].split(".")[1]), input_array)

        # test_output = c_model(torch.tensor(input_array/100.).float())
        # print(f"test_Outputs: ", test_output)


        # Receive activation from client
        # outputs = device.readline().decode()

        if only_forward:
            # Perform server-side computation (forward)
            # hidden_activation = convert_string_to_array(outputs)
            label = convert_string_to_array(str(num_button), one_hot = True)
            # print(f"input: ", input_array)
            print(f"label: ", num_button)
            forward_accu, forward_error = server_compute(s_model, s_optimizer, input_array, label, only_forward= True)
        else:
            # Receive label from client
            nb = device.readline()[:-2]
            # print(str(nb))
            # Perform server-side computation (forward/backward)
            # hidden_activation = convert_string_to_array(outputs)
            label = convert_string_to_array(str(nb), one_hot = True)
            # print(f"input: ", input_array)
            print(f"label: ", label)
            forward_accu, forward_error = server_compute(s_model, s_optimizer, input_array, label, only_forward= False)
            
            # Send Error Array to client to continue backward #TODO: implement this
            # for i in range(size_hidden_nodes): # hidden layer
            #     d.read() # wait until confirmatio
            #     float_num = error_array[i]
            #     data = struct.pack('f', float_num)
            #     d.write(data)

            # sendmodel_confirmation = d.readline().decoder()
            # print(f'Model sent confirmation: {sendmodel_confirmation}')
            
            # if (forward_error > 0.28):
            #     print(f"[{device.port}] Sample {samplePath} generated an error of {forward_error}")

        # print(f"Fordward millis received: ", device.readline().decode())
        # print(f"Backward millis received: ", device.readline().decode())
        device.readline().decode() # Accept 'Done' command

        ne = device.readline()[:-2]

        n_epooch = int(ne)
        running_batch_accu += forward_accu
        graph.append([n_epooch, forward_error, deviceIndex])
        print(f"Error: ", forward_error)

def sendTestSamples(device, deviceIndex):
    global test_mountains

    print(f"[{device.port}] Sending test samples from {0} to {60}")

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
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        sendSample(device, 'datasets/mountains/'+filename, num_button, deviceIndex, True)




# def read_graph(device, deviceIndex):
#     global repaint_graph

#     outputs = device.readline().decode()
#     print(f"Outputs: ", outputs)

#     error = device.readline().decode()
#     print(f"Error: ", error)

#     ne = device.readline()[:-2]
#     n_epooch = int(ne)

#     n_error = device.read(4)
#     [n_error] = struct.unpack('f', n_error)
#     nb = device.readline()[:-2]
#     graph.append([n_epooch, n_error, deviceIndex])
#     repaint_graph = True
#     return n_error

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
    d.reset_input_buffer()
    d.reset_output_buffer()
    d.timeout = 5

    print(f'Starting connection to {d.port} ...') # Hanshake
    d.write(b'>') # Python --> SYN --> Arduino
    if d.read() == b'<': # Python <-- SYN ACK <-- Arduino
        d.write(b's') # Python --> ACK --> Arduino
        
        print('Connection accepted.')
        devices_connected.append(d)
        #devices_hidden_layer = np.vstack((devices_hidden_layer, np.empty(size_hidden_layer)))
        #devices_output_layer = np.vstack((devices_output_layer, np.empty(size_output_layer)))
        d.timeout = None

        print_until_keyword('start', d)
        devices_num_epochs.append(int(d.readline()[:-2]))

        print(f'Receiving model from {d.port} ...')
        ini_time = time.time()

        for i in range(size_hidden_layer): # hidden layer
            data = d.read(4)
            [float_num] = struct.unpack('f', data)
            devices_hidden_layer[device_index][i] = float_num

        # for i in range(size_output_layer): # output layer
        #     data = d.read(4)
        #     [float_num] = struct.unpack('f', data)
        #     devices_output_layer[device_index][i] = float_num

        print(f'Model received from {d.port} ({time.time()-ini_time} seconds)')

        # if it was not connected before, we dont use the devices' model
        if not d in old_devices_connected:
            devices_num_epochs[device_index] = 0
            print(f'Model not used. The device {d.port} has an outdated model')

    else:
        print(f'Connection timed out. Skipping {d.port}.')

def sendModel(d, hidden_layer, output_layer):
    ini_time = time.time()
    for i in range(size_hidden_layer): # hidden layer
        d.read() # wait until confirmatio
        float_num = hidden_layer[i]
        data = struct.pack('f', float_num)
        d.write(data)

    # for i in range(size_output_layer): # output layer
    #     d.read() # wait until confirmatio
    #     float_num = output_layer[i]
    #     data = struct.pack('f', float_num)
    #     d.write(data)

    # sendmodel_confirmation = d.readline().decoder()
    # print(f'Model sent confirmation: {sendmodel_confirmation}')
    print(f'Model sent to {d.port} ({time.time()-ini_time} seconds)')

def startFL():
    global devices_connected, hidden_layer, output_layer, pauseListen

    pauseListen = True

    print('Starting Federated Learning')
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
        print(f'Average millis: {(time.time()*1000)-ini_time} milliseconds)')

    # Doing validation
    # c_model.load_state_dict({'client.0.weight': torch.tensor(hidden_layer[:size_hidden_nodes*650]).view(650, size_hidden_nodes).t().float(), 'client.0.bias': torch.tensor(hidden_layer[size_hidden_nodes*650:]).float()})
    
    # test_in, test_out = getSamplesIID(50, 200)

    # error, accu = server_validate(test_in, test_out)
    # print("======Testing Start======")
    # print(f"==Error {error}==")
    # print(f"==Accuracy {accu}==")
    # val_graph.append([error, accu, 0])
    # print("======Testing End======")

    #################
    # Sending models
    #################
    threads = []
    for d in devices_connected:
        print(f'Sending model to {d.port} ...')

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



# To load a Pre-trained model
# hidden_layer = np.load("./hidden_montserrat.npy")
# output_layer = np.load("./output_montserrat.npy")



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
epoch_size = 3 # default = 1
for _ in range(epoch_size):
    
    for batch in range(int(samples_per_device/batch_size)):
        running_batch_accu = 0
        for deviceIndex, device in enumerate(devices):
            if experiment == 'iid' or experiment == 'train-test':
                thread = threading.Thread(target=sendSamplesIID, args=(device, deviceIndex, batch_size, batch))
            elif experiment == 'no-iid':
                thread = threading.Thread(target=sendSamplesNonIID, args=(device, deviceIndex, batch_size, batch))
            elif experiment == 'custom':
                thread = threading.Thread(target=sendSamplesIIDCustom, args=(device, deviceIndex, batch_size, batch))

            thread.daemon = True
            thread.start()
            threads.append(thread)
        for thread in threads: thread.join() # Wait for all the threads to end
        startFL()
        print("This round accuracy is {}.".format(running_batch_accu/batch_size))
        running_batch_accu_list.append(running_batch_accu/batch_size)
    
train_time = time.time()-ini_time
# print(f'Trained in ({train_time} seconds)')

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
# plt.title(f"Loss vs Epoch")
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
figname = f"newplots/ES{epoch_size}-BS{eq_batch_size}-LR{learningRate}-M{momentum}-NH{number_hidden}-HS{hidden_size}-HN{size_hidden_nodes}-TT{train_time}-{experiment}_train.eps"
plt.savefig(figname, format='eps')
print(f"Generated {figname}")

# plt.figure(2)
# plt.ion()
# plt.show()
# plt.rc('font', size=font_sm)          # controls default text sizes
# plt.rc('axes', titlesize=font_sm)     # fontsize of the axes title
# plt.rc('axes', labelsize=font_md)     # fontsize of the x and y labels
# plt.rc('xtick', labelsize=font_sm)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=font_sm)    # fontsize of the tick labels
# plt.rc('legend', fontsize=font_sm)    # legend fontsize
# plt.rc('figure', titlesize=font_xl)   # fontsize of the figure title
# plot_val_graph()
# figname2 = f"newplots/ES{epoch_size}-BS{batch_size}-LR{learningRate}-M{momentum}-NH{number_hidden}-HS{hidden_size}-HN{size_hidden_nodes}-TT{train_time}-{experiment}_val.eps"
# plt.savefig(figname2, format='eps')
# print(f"Generated {figname2}")

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
figname3 = f"newplots/ES{epoch_size}-BS{eq_batch_size}-LR{learningRate}-M{momentum}-NH{number_hidden}-HS{hidden_size}-HN{size_hidden_nodes}-TT{train_time}-{experiment}_train_accu.eps"
plt.savefig(figname3, format='eps')
print(f"Generated {figname3}")