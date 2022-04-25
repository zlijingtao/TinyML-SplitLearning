# %%
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from collections import OrderedDict
import tensorflow as tf
from torch.autograd import Variable
from onnx_tf.backend import prepare

from models import server_conv2d_model

# class MLP(nn.Module):
#     def __init__(self, input_dims, n_hiddens, n_class):
#         super(MLP, self).__init__()
#         assert isinstance(input_dims, int), 'Please provide int for input_dims'
#         self.input_dims = input_dims
#         current_dims = input_dims
#         layers = OrderedDict()

#         if isinstance(n_hiddens, int):
#             n_hiddens = [n_hiddens]
#         else:
#             n_hiddens = list(n_hiddens)
#         for i, n_hidden in enumerate(n_hiddens):
#             layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
#             layers['relu{}'.format(i+1)] = nn.ReLU()
#             layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
#             current_dims = n_hidden
#         layers['out'] = nn.Linear(current_dims, n_class)

#         self.model= nn.Sequential(layers)
#         print(self.model)

#     def forward(self, input):
#         input = input.view(input.size(0), -1)
#         assert input.size(1) == self.input_dims
#         return self.model.forward(input)

# print("%s" % sys.argv[1])
# print("%s" % sys.argv[2])


# Load the trained model from file
# trained_dict = torch.load(sys.argv[1], map_location={'cuda:0': 'cpu'})
# exit()
def pytorch2tflite(torch_model,model_name):
    # trained_dict = torch.load("s_model.pth")
    trained_dict = torch.load(torch_model)
    print("load succeed!")
    trained_model = server_conv2d_model(7, 1, 128, (50,13))
    trained_model.load_state_dict(trained_dict)
    # exit()
    if not os.path.exists("tfl_model"):
        os.makedirs("tfl_model")

    # Export the trained model to ONNX
    dummy_input = Variable(torch.randn(1, 12, 6,24)) # one black and white 28 x 28 picture will be the input to the model
    torch.onnx.export(trained_model, dummy_input, f"tfl_model/{model_name}.onnx")

    # Load the ONNX file
    model = onnx.load(f"tfl_model/{model_name}.onnx")
    ## verify onnx model
    onnx.checker.check_model(model)
    # print("onnx model checked")
    # print(onnx.helper.printable_graph(model.graph))

    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model)

    #%%
    # test tf model
    # output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
    # print('The digit is classified as ', np.argmax(output))

    # Input nodes to the model
    # print('inputs:', tf_rep.inputs)

    # Output nodes from the model
    # print('outputs:', tf_rep.outputs)

    # All nodes in the model
    # print('tensor_dict:')
    # print(tf_rep.tensor_dict)

    tf_rep.export_graph(f"tfl_model/{model_name}.pb")

    # converter = tf.lite.TFLiteConverter.from_frozen_graph(
    #  f       "%s/{model_name}.pb" % sys.argv[1], tf_rep.inputs, tf_rep.outputs)
    # --------- above not work because "from_frozen_graph" is in older tf version
    converter = tf.lite.TFLiteConverter.from_saved_model(f"tfl_model/{model_name}.pb")
    tflite_model = converter.convert()
    open(f"tfl_model/{model_name}.tflite", "wb").write(tflite_model)
    print("------ finished: from tf to tfl ------------")