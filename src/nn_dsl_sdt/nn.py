import abc
import sys

import torch.nn as nn
import hiddenlayer as hl
from typing import Any


class Container():

    __NEXT_ID = 0

    def __init__(self):
        print("FuncIn: Init Container", file=sys.stderr)
        Container.__NEXT_ID += 1
        self.id = Container.__NEXT_ID
        self.data = None

    def write(self, data): self.data = data

    def read(self): return self.data

    def getId(self): return self.id


class Storage:

    def __init__(self):
        print("FuncIn: Init Storage", file=sys.stderr)

        self.inputContainer = Container()
        self.outputContainer = self.inputContainer
        self.data = {}
        self.data[self.inputContainer.getId()] = self.inputContainer

    def get_container(self, id:int, isInput:bool=False, isOutput:bool=False):
        if isInput: return self.inputContainer
        if isOutput: return self.outputContainer
        if id not in self.data: return None
        return self.data[id]

    def add_container(self, setOutput=True) -> Container:
        cont = Container()
        self.data[cont.getId()] = cont
        if setOutput: self.outputContainer = cont
        return cont

    def inputId(self): return self.inputContainer.getId()

    def outputId(self): return self.outputContainer.getId()

    def pprint(self):
        print("Input container ", self.inputContainer.getId())
        print("Output container", self.outputContainer.getId())

class Node:

    def __init__(self, funct, res:int, args=[], storage:Storage=None):
        print("FuncIn: Init Node", file=sys.stderr)
        assert funct is not None, "Can not make Node with no function"
        assert storage is not None, "Can not make Node with no storage"

        self.funct = funct
        self.args = args
        self.res = res
        self.storage = storage

    def eval(self): self.res.write(self.funct([self.storage.get_container(i).read() for i in self.args]))

    def pprint(self):
        print("Node from {} to {}".format(self.args, self.res.getId()))


# Format AutoML Iteratable Neural Network 
class NeuralNetwork():

    def __init__(self):
        print("FuncIn: Init NeuralNetwork", file=sys.stderr)

        self.storage = Storage()
        self.nodes = []

    def add_layer(self, function, input_layers=[]) -> int:
        res = self.storage.add_container()
        if len(input_layers) == 0:
            input_layers = [self.storage.inputId()]
        node = Node(function, res, input_layers, self.storage)
        self.nodes.append(node)
        return res.getId()

    def add_var(self, data) -> int:
        res = self.storage.add_container(False)
        res.write(data)
        return res.getId()

    def from_net(self, net: nn.Module, arg: Any):        
        h = hl.build_graph(net, arg)
        return

    def eval(self, data):
        self.storage.get_container(self.storage.inputId()).write(data)
        for i in self.nodes: i.eval()
        return self.storage.get_container(self.storage.outputId()).read()

    def pprint(self):
        print("Print Neural Network:")
        self.storage.pprint()
        for i in self.nodes: i.pprint()


class FAMLINN(NeuralNetwork):
    pass