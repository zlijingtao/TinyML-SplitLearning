import torch
import torch.nn as nn
import torch.nn.init as init


class client_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, hidden_size = 25):
        super(client_model, self).__init__()

        model_list = []
        
        model_list.append(nn.Linear(650, hidden_size, bias = True))
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
        model_list.append(nn.Conv1d(13, 16, kernel_size = 5, padding="same"))
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
        model_list.append(nn.Conv1d(16, 8, kernel_size = 5, padding="same"))
        model_list.append(nn.BatchNorm1d(8))
        model_list.append(nn.ReLU())
        # model_list.append(nn.Conv1d(8, 4, kernel_size = 3, padding="same"))
        # model_list.append(nn.BatchNorm1d(4))
        # model_list.append(nn.ReLU())
        model_list.append(nn.Flatten(1))
        last_layer_input_size = 400
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

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self,x):
        x = x.view(-1, 1, 13, 50)
        return x
class client_conv2d_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self):
        super(client_conv2d_model, self).__init__()

        model_list = []
        # model_list.append(Reshape())
        model_list.append(nn.Conv2d(1, 12, kernel_size = 3, stride = 2, bias = False))
        model_list.append(nn.ReLU())

        self.client = nn.Sequential(*model_list)

    def forward(self, x):
        x = x.view(-1, 1, 13, 50)
        out = self.client(x)
        return out

class server_conv2d_model(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, num_class = 3, number_hidden = 0, hidden_size = 128, input_size = (50, 13)):
        super(server_conv2d_model, self).__init__()

        # last_layer_input_size = input_size
        model_list = []
        model_list.append(nn.Conv2d(12,30, kernel_size = 3, stride = 2, bias = False))
        model_list.append(nn.BatchNorm2d(30))
        model_list.append(nn.ReLU())
        model_list.append(nn.Flatten(1))
        last_layer_input_size = 660
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