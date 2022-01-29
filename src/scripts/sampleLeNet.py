from nn_dsl_sdt.nn import *
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np

# PyTorch: original
layers = [
    nn.Conv2d(1, 32, 3, 1),
    nn.Conv2d(32, 64, 3, 1),
    nn.Dropout2d(0.25),
    nn.Dropout2d(0.5),
    nn.Linear(9216, 128),
    nn.Linear(128, 10),
] 

class PT_Net(nn.Module):
    def __init__(self):
      super(PT_Net, self).__init__()
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

def trainTheNet(net):
    import torch, torchvision
    from torchvision import datasets, transforms
    from torch import nn, optim
    from torch.nn import functional as F

    import numpy as np
    import shap

    batch_size = 128
    num_epochs = 2
    device = torch.device('cpu')

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output.log(), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    model = net.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # since shuffle=True, this is a random sample of test data
    batch = next(iter(test_loader))
    images, _ = batch

    background = images[:100]
    test_images = images[100:103]

    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)



def sample_le_net():
    print("LeNet example")
    
    for i in layers:
        # i.reset_parameters()
        print(list(i.parameters()))

    originalNet = PT_Net()
    
    # trainTheNet(originalNet)

    # NeuralNetwork: generate net
    
    n = NeuralNetwork()
    arg = n.add_layer(lambda x: x[0])
    x = n.add_layer(lambda x: layers[0](x[0]), [arg])
    x = n.add_layer(lambda x: nnF.relu(x[0]), [x])
    x = n.add_layer(lambda x: layers[1](x[0]), [x])
    x = n.add_layer(lambda x: nnF.relu(x[0]), [x])
    x = n.add_layer(lambda x: nnF.max_pool2d(x[0], 2), [x])
    x = n.add_layer(lambda x: torch.flatten(x[0], 1), [x])
    x = n.add_layer(lambda x: layers[4](x[0]), [x])
    x = n.add_layer(lambda x: nnF.relu(x[0]), [x])
    x = n.add_layer(lambda x: layers[5](x[0]), [x])
    x = n.add_layer(lambda x: nnF.log_softmax(x[0], dim=1), [x])

    image = torch.rand((1, 1, 28, 28))
    print(n.eval(image))
    print(originalNet(image))
    
    print(n.eval(image))
    print(originalNet(image))
    

################################################################################

def sampleTrainSomeNet():
    import torch, torchvision
    from torchvision import datasets, transforms
    from torch import nn, optim
    from torch.nn import functional as F

    import numpy as np
    import shap


    batch_size = 128
    num_epochs = 2
    device = torch.device('cpu')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.Dropout(),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(320, 50),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(50, 10),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)
            return x

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output.log(), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # since shuffle=True, this is a random sample of test data
    batch = next(iter(test_loader))
    images, _ = batch

    background = images[:100]
    test_images = images[100:103]

    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    # plot the feature attributions
    shap.image_plot(shap_numpy, -test_numpy)


################################################################################


sampleMnistLayers = [
    nn.Conv2d(1, 10, kernel_size=5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.Dropout(),
    nn.MaxPool2d(2),
    nn.ReLU(),    
        
    nn.Linear(320, 50),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(50, 10),
    nn.Softmax(dim=1)
]

def sampleMnist():
    import torch, torchvision
    from torchvision import datasets, transforms
    from torch import nn, optim
    from torch.nn import functional as F

    import numpy as np
    import shap


    batch_size = 128
    num_epochs = 2
    device = torch.device('cpu')


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv_layers = nn.Sequential(*sampleMnistLayers[0:7])
            self.fc_layers = nn.Sequential(*sampleMnistLayers[7:12])

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)
            return x

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output.log(), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # since shuffle=True, this is a random sample of test data
    batch = next(iter(test_loader))
    images, _ = batch

    background = images[:100]
    test_images = images[100:103]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    
    # NeuralNetwork: generate net
    
    #n = NeuralNetwork()
    #arg = n.add_layer(lambda x: x[0])
    #x = n.add_layer(lambda x: sampleMnistLayers[0](x[0]), [arg])
    #x = n.add_layer(lambda x: sampleMnistLayers[1](x[0]), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[2](x[0]), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[3](x[0]), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[4](x[0]), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[5](x[0]), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[6](x[0]), [x])
    #x = n.add_layer(lambda x: x[0].view(-1, 320), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[7](x[0]), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[8](x[0]), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[9](x[0]), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[10](x[0]), [x])
    #x = n.add_layer(lambda x: sampleMnistLayers[11](x[0]), [x])
   
    n = NeuralNetwork()
    arg = n.add_layer(lambda x: x[0])
    x = n.add_layer(lambda x: nn.Sequential(*sampleMnistLayers[0:7])(x[0]), [arg])
    x = n.add_layer(lambda x: x[0].view(-1, 320), [x])
    x = n.add_layer(lambda x: nn.Sequential(*sampleMnistLayers[7:12])(x[0]), [x])
    
    #image = [np.newaxis, test_images[0]]
    print(model(test_images))
    print(n.eval(image))

################################################################################


def main():
    sample_le_net()
    sampleMnist()

main()