from nn_dsl_sdt.nn import *
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np

class ONet(nn.Module):
    def __init__(self):
      super(ONet, self).__init__()
      layers = [
        nn.Conv2d(1, 32, 3, 1),
        nn.Conv2d(32, 64, 3, 1),
        nn.Dropout2d(0.25),
        nn.Dropout2d(0.5),
        nn.Linear(9216, 128),
        nn.Linear(128, 10),
      ] 
      self.conv1 = layers[0]
      self.conv2 = layers[1]
      self.fc1 = layers[4]
      self.fc2 = layers[5]

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = nnF.relu(x)

      x = self.conv2(x)
      x = nnF.relu(x)

      # Run max pooling over x
      x = nnF.max_pool2d(x, 2)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = nnF.relu(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = nnF.log_softmax(x, dim=1)
      return output


def main():
    image = torch.rand((1, 1, 28, 28))
    originalNet = ONet()
    import torch.onnx.utils as ptonnxut
    import torch.onnx as ptonnx
    # ptonnx.export
    graph, x, y = ptonnxut._model_to_graph(originalNet, image)
    print(originalNet(image))
    print(graph)



main()