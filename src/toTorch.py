from nn_dsl_sdt.nn import *

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
import torch.onnx
import onnx
import onnxruntime

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            #nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x

def sampleData():
    return torch.rand((1, 1, 28, 28))

def convertExample():
    print("Convert example: ")
    n = Net()
    # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    torch.onnx.export(n,               # model being run
                  sampleData(),                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    arg = sampleData()
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(arg)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("Original: ", n(arg))
    print("ONNX_converted: ", ort_outs)

    for node in onnx_model.graph.node:
        print(node)
        print("--------------------------------")

    # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    # https://pythonrepo.com/repo/fumihwh-onnx-pytorch
    # https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
    # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    

if __name__ == '__main__':
    convertExample()