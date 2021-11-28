from nn_dsl_sdt.nn import *
import numpy as np

def sample_xyz():
    print("Net: (x + y) * z")
    n = NeuralNetwork()
    x_op = n.add_layer(lambda x: x[0])
    y = n.add_var(4)
    z = n.add_var(1.5)
    y_op = n.add_layer(lambda x: x[0] + x[1], [x_op, y])
    z_op = n.add_layer(lambda x: x[0] * x[1], [y_op, z])
    n.pprint()
    print("Net(1): ", n.eval(1))
    print("Net(2): ", n.eval(2))
   

def sample_iris():
    print("Net: X -> W1 -> sigm -> W2 -> Y")
    HIDDEN_NODES = 10
    DATA = np.random.rand(4, 2).reshape((2, 4))
    n = NeuralNetwork()
    X = n.add_layer(lambda x: x[0])
    W1 = n.add_var(np.random.rand(4, HIDDEN_NODES).reshape((4, HIDDEN_NODES)))
    W2 = n.add_var(np.random.rand(HIDDEN_NODES, 3).reshape((HIDDEN_NODES, 3)))
    #A1 = np.sigmoid(np.matmul(X, W1))
    xw = n.add_layer(lambda x: np.matmul(x[0], x[1]), [X, W1])
    a1 = n.add_layer(lambda x: 1 / (1 + np.exp(-x[0])), [xw])
    #y_est = np.sigmoid(np.matmul(A1, W2))
    aw = n.add_layer(lambda x: np.matmul(x[0], x[1]), [a1, W2])
    y = n.add_layer(lambda x: 1 / (1 + np.exp(-x[0])), [aw])
    n.pprint()
    print("Net(): ")
    print(n.eval(DATA))

def main():
    sample_xyz()
    sample_iris()

main()