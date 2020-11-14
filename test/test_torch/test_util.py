# https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py
def torch_simple_example():
    from torch.nn import Sequential, Conv2d, Linear, ReLU, MaxPool2d, Dropout2d, Softmax, CrossEntropyLoss
    from torch.optim import Adam
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import transforms

    log_interval = 103

    class Flatten(torch.nn.Module):
        __constants__ = ['start_dim', 'end_dim']

        def __init__(self, start_dim=1, end_dim=-1):
            super(Flatten, self).__init__()
            self.start_dim = start_dim
            self.end_dim = end_dim

        def forward(self, input: torch.Tensor):
            return input.flatten(self.start_dim, self.end_dim)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
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
                test_loss += loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    # Set the random seed
    torch.manual_seed(0)
    device = torch.device('cpu')

    # Load Mnist Dataset
    train_mnist = MNIST('data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_mnist = MNIST('data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    # N is batch size;
    N, epochs = 500, 1

    # Training Loader
    train_loader = DataLoader(train_mnist, batch_size=N)

    # Testing Loader
    test_loader = DataLoader(test_mnist, batch_size=N)

    # Create random Tensors to hold inputs and outputs
    # x = torch.randn(N, D_in, device=device)
    # y = torch.randn(N, D_out, device=device)

    # Use the nn package to define our model as a sequence of layers
    model = Sequential(
        Conv2d(1, 32, 3, 1),
        ReLU(),
        Conv2d(32, 64, 3, 1),
        MaxPool2d(2),
        Dropout2d(0.25),
        Flatten(),
        Linear(9216, 128),
        ReLU(),
        Dropout2d(0.5),
        Linear(128, 10),
        Softmax(dim=1)
    ).to(device)

    # Define loss function
    loss_fn = CrossEntropyLoss().to(device)

    # define the optimize
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch)
        test(model=model, device=device, test_loader=test_loader)


def data_transform(data, sample_shape):
    import numpy as np
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(data[i].reshape(sample_shape))
    return np.asarray(data_t, dtype=np.float32)


def torch_3d_mnist_example(tracker, train_data, test_data):
    import torch
    from torch import nn
    import h5py
    import numpy as np

    @tracker.decorators.dataset(name="3D-MNIST", target_columns=[-1])
    def load_3d_mnist(path):
        """
        The aim of this dataset is to provide a simple way to get started with 3D computer vision problems such as 3D shape recognition.

        Accurate 3D point clouds can (easily and cheaply) be adquired nowdays from different sources:

         - RGB-D devices: Google Tango, Microsoft Kinect, etc.
         - Lidar.
         - 3D reconstruction from multiple images.

        However there is a lack of large 3D datasets (you can find a good one here based on triangular meshes); it's especially hard to find datasets based on point clouds (wich is the raw output from every 3D sensing device).

        This dataset contains 3D point clouds generated from the original images of the MNIST dataset to bring a familiar introduction to 3D to people used to work with 2D datasets (images).

        The full dataset is splitted into arrays:

        X_train (10000, 4096)
        y_train (10000)
        X_test(2000, 4096)
        y_test (2000)

        """

        with h5py.File(path, "r") as hf:
            X_train, y_train = hf["X_train"][:], hf["y_train"][:]
            X_test, y_test = hf["X_test"][:], hf["y_test"][:]
            train_data = np.concatenate([X_train, y_train.reshape(len(y_train), 1)], axis=1)
            test_data = np.concatenate([X_test, y_test.reshape(len(y_test), 1)], axis=1)
            data = np.concatenate([train_data, test_data], axis=0)
        return data

    class CNNModel(nn.Module):
        def __init__(self, dim_output):
            super(CNNModel, self).__init__()

            self.conv_layer1 = self._conv_layer_set(3, 32)
            self.conv_layer2 = self._conv_layer_set(32, 64)
            self.fc1 = nn.Linear(2 ** 3 * 64, 128)
            self.fc2 = nn.Linear(128, dim_output)
            self.relu = nn.LeakyReLU()
            self.batch = nn.BatchNorm1d(128)
            self.drop = nn.Dropout(p=0.15)

        def _conv_layer_set(self, in_c, out_c):
            conv_layer = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
                nn.LeakyReLU(),
                nn.MaxPool3d((2, 2, 2)),
            )
            return conv_layer

        def forward(self, x):
            # Set 1
            out = self.conv_layer1(x)
            out = self.conv_layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.batch(out)
            out = self.drop(out)
            out = self.fc2(out)

            return out

    # Sample shape
    sample_shape = (16, 16, 16)

    # Load 3d Mnist data
    path = "/home/mehdi/Desktop/Workspace/Padre_project/PyPadre/pypads-examples/Notebooks-DataScience Lab/data/" \
           "3d-mnist/full_dataset_vectors.h5"
    data = load_3d_mnist(path)
    X_train, y_train = data[:10000, :-1], data[:10000, -1]
    X_test, y_test = data[:10000, :-1], data[:10000, -1]

    # Reshape data into 3D format (16,16,16)
    X_train = data_transform(X_train, sample_shape)
    X_test = data_transform(X_test, sample_shape)

    train_x = torch.from_numpy(X_train).float()
    train_y = torch.from_numpy(y_train).long()
    test_x = torch.from_numpy(X_test).float()
    test_y = torch.from_numpy(y_test).long()

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)


    # Definition of hyperparameters
    batch_size = 100
    n_iters = 4500
    num_epochs = n_iters / (len(train_x) / batch_size)
    num_epochs = int(num_epochs)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    # Create CNN
    model = CNNModel(10)
    # model.cuda()
    print(model)

    # Cross Entropy Loss
    error = nn.CrossEntropyLoss()

    # SGD Optimizer
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
