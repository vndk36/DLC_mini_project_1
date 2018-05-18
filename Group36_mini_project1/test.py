
import torch
import numpy
import os
import errno

from torch import optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


from six.moves import urllib


def tensor_from_file(root, filename,
                     base_url = 'https://documents.epfl.ch/users/f/fl/fleuret/www/data/bci'):

    file_path = os.path.join(root, filename)

    if not os.path.exists(file_path):
        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = base_url + '/' + filename

        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        with open(file_path, 'wb') as f:
            f.write(data.read())

    return torch.from_numpy(numpy.loadtxt(file_path))


def load(root, train = True, download = True, one_khz = False):


    """
    Args:

        root (string): Root directory of dataset.

        train (bool, optional): If True, creates dataset from training data.

        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

        one_khz (bool, optional): If True, creates dataset from the 1000Hz data instead
            of the default 100Hz.

    """

    nb_electrodes = 28

    if train:

        if one_khz:
            dataset = tensor_from_file(root, 'sp1s_aa_train_1000Hz.txt')
        else:
            dataset = tensor_from_file(root, 'sp1s_aa_train.txt')

        input = dataset.narrow(1, 1, dataset.size(1) - 1)
        input = input.float().view(input.size(0), nb_electrodes, -1)
        target = dataset.narrow(1, 0, 1).clone().view(-1).long()  # changer le type suivant le loss criterion

    else:

        if one_khz:
            input = tensor_from_file(root, 'sp1s_aa_test_1000Hz.txt')
        else:
            input = tensor_from_file(root, 'sp1s_aa_test.txt')
        target = tensor_from_file(root, 'labels_data_set_iv.txt')

        input = input.float().view(input.size(0), nb_electrodes, -1)
        target = target.view(-1).long()  # changer le type suivant le loss criterion

    return input, target

# END OF THE GIVEN CODE
########################################################################################################################

# Basic Computation of the number of error of a model on a given data-set

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0
    model.eval()
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(0, mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


########################################################################################################################
# Loading data, Increase the Dataset and Create Validation Set

def valSet(train_input, train_target ,sizeVal):
    # Goal: create a validation set from the training set
    # divides randomly the matrix "train_input" and its associated target in two.
    # Output is a new "train_input" (of the size (train_input.size(0)-sizeVal)xtrain_input.size(1)xtrain_input.size(2))
    # and a "validation_set" whose first dimension has a size of sizeVal
    
    permutation = torch.randperm(train_input.size()[0])
    validation_set_input= train_input[permutation].narrow(0,0,sizeVal )
    validation_set_target= train_target[permutation].narrow(0,0,sizeVal)
    new_train_input= train_input[permutation].narrow(0,sizeVal, train_input .size()[0]-sizeVal)
    new_train_target= train_target[permutation].narrow(0,sizeVal, train_input. size()[0]-sizeVal)
    
    return new_train_input, new_train_target, validation_set_input, validation_set_target


def load_data():
    # Goal: download (if necessary) the dataset and separate randomly a validation set from the training set.
    # Create"fake data" to be added to the train set. Demean and set the standard deviation
    # of the different dataset created to 1

    print("Loading new validation set")
    train_input, train_target = load("data", True, False)
    test_input, test_target = load("data", False, False)
    train_input, train_target, test_input, test_target = Variable(train_input), Variable(train_target), \
                                                         Variable(test_input), Variable(test_target)

    train_input, train_target, validation_set_input, validation_set_target = valSet(train_input, train_target, 16)

    # adding white noise to the signal
    dataNoisy= Variable(torch.Tensor(train_input.size()))
    for i in range(0, train_input.size(0)):
        for j in range(0, train_input.size(1)):
            noise = numpy.random.normal(0,5,50 )  # vector of 50 values with normal distribution of mean=0, std=5
            # noise = numpy.random.uniform(-2,2, 50)
            noiseVar= Variable(torch.Tensor(noise))
            dataNoisy[i,j,:]= train_input[i,j,:].add ( noiseVar)

    # adding white noise to the signal
    dataNoisy2= Variable (torch.Tensor(train_input.size()))
    for i in range(0, train_input.size(0)):
        for j in range(0, train_input.size(1)):
            noise = numpy.random.normal(0,3,50)  # vector of 50 values with normal distribution of mean=0, std=3
            # noise = numpy.random.uniform(-4,4,50)
            noiseVar= Variable (torch.Tensor(noise))
            dataNoisy2[i,j,:]= train_input[i,j,:].add( noiseVar)

    train_input = torch.cat((train_input, dataNoisy), 0)  # the two noisy matrices  are concatenated with the real one
    train_input = torch.cat((train_input, dataNoisy2), 0)

    new_train_target=torch.cat(( train_target, train_target), 0) # increase of the target size also
    train_target=torch.cat(( new_train_target, train_target), 0)


    # Demean of the data and removing the std
    train_input = train_input.sub_(train_input.data.mean()).div_(train_input.data.std()) 
    test_input = test_input.sub_(test_input.data.mean()).div_(test_input.data.std())
    validation_set_input = validation_set_input.sub_(validation_set_input.data.mean()).div_(
        validation_set_input.data.std())
    
    return train_input, train_target, test_input, test_target, validation_set_input, validation_set_target

########################################################################################################################
# # Train Model function



def train_model(model, train_input, train_target, loss_plot, nb_epochs, batch_size, lr):
    # Goal: train a model with a chosen network on the input dataset

    # Set the step size to the desired one for Lr update
    step_size = 20
    model.train()

    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(model.parameters(), lr = lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optimizer to update the Learning rate
    scheduler = 0

    # Activate (uncomment) one of the two step optimizer if you want to have one

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, threshold=1, factor=0.5, patience = 30 )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma=0.1, last_epoch=-1)


    for e in range(0, nb_epochs):
        # Randomization of the batch size
        permutation = torch.randperm(train_input.size()[0])
        sum_loss = 0
        for b in range(0, train_input.size(0), batch_size):
            # Permutation to choose the indexes of the data in the future batch
            indices = permutation[b:b + batch_size]
            # The data associated to the indexes stored inside "indices" form the batch
            batch_input, batch_target = train_input[indices], train_target[indices]

            output = model(batch_input)
            loss = criterion(output, batch_target)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss = sum_loss + loss

        # print("The loss is :"+str(loss)+" for epoch :"+str(e))
        loss_plot.append(sum_loss.data.numpy())
        # print(str(int((e/nb_epochs)*100))+"%")

        # Function to exit the learning function if the loss goes close enough to zero
        if sum_loss.data.numpy() < 0.01:
            break
        # If a scheduler is defined, this function will update the scheduler, PLEASE SELECT THE RIGHT ONE
        if scheduler != 0:
            # scheduler.step(sum_loss.data.numpy())
            scheduler.step()

    return e



def train_and_test_model(model, nb_epochs, batch_size, lr, nb_training):
    train_error = []
    validation_error = []
    test_error = []

    for i in range(0, nb_training):
        train_input, train_target, test_input, test_target, validation_set_input, validation_set_target = load_data()
        print("Training the model, please wait...")
        for p in model.parameters(): p.data.normal_(0, 0.01)
        loss_plot = []
        epochs_reached = train_model(model, train_input, train_target, loss_plot, nb_epochs, batch_size=batch_size,
                                     lr=lr)

        train_error.append(compute_nb_errors(model, train_input, train_target, 10) / train_input.size(0) * 100)
        validation_error.append(
            compute_nb_errors(model, validation_set_input, validation_set_target, 16) / validation_set_input.size(
                0) * 100)
        test_error.append(compute_nb_errors(model, test_input, test_target, 50) / test_input.size(0) * 100)

        print('train_error {:.02f}% validation_set_error {:.02f}% test_error {:.02f}%'.format(train_error[i],
                                                                                              validation_error[i],
                                                                                              test_error[i]))
        for p in model.parameters(): p.data.normal_(0, 0.01)
        plt.plot(range(epochs_reached + 1), loss_plot)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        print('train_error AVG {:.02f}% validation_set_error AVG {:.02f}% test_error AVG {:.02f}%'.format(
            numpy.asarray(train_error).mean(),
            numpy.asarray(validation_error).mean(),
            numpy.asarray(test_error).mean()))
        print('train_error BEST {:.02f}% validation_set_error BEST {:.02f}% test_error BEST {:.02f}%\n\n\n'.format(
            numpy.asarray(train_error).min(),
            numpy.asarray(validation_error).min(),
            numpy.asarray(test_error).min()))


#######################################################################################################################
# Network Models

# HERE ALL THE MODELS WILL BE LOADED. IF YOU WANT SOME OF THEM DISABLE, PLEASE COMMENT THEM. OTHERWISE LET THEM HERE

##########


class Net_lin(nn.Module):
    def __init__(self, nb_hidden):
        super(Net_lin, self).__init__()
        self.fc1 = nn.Linear(28 * 50, 1000)
        self.fc2 = nn.Linear(1000, 400)
        self.fc3 = nn.Linear(400, 20)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, x.data.size()[1] * x.data.size()[2])))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

##########


class Net_lin_dropout(nn.Module):
    def __init__(self, nb_hidden):
        super(Net_lin_dropout, self).__init__()
        self.fc1 = nn.Linear(28 * 50, 1000)
        self.fc2 = nn.Linear(1000, 400)
        self.fc3 = nn.Linear(400, 20)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, x.data.size()[1] * x.data.size()[2])))
        x = F.relu(self.fc2(x))
        nn.Dropout()
        x = F.relu(self.fc3(x))
        nn.Dropout()
        x = self.fc4(x)
        return x


##########


class NetDROP(nn.Module):
    def __init__(self, nb_hidden):
        super(NetDROP, self).__init__()
        self.conv1 = nn.Conv1d(28, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4)
        self.fc1 = nn.Linear(256 * 4, 3 * nb_hidden)

        self.fc2 = nn.Linear(nb_hidden * 3, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv3(x), kernel_size=2, stride=2))
        x = self.dropout(x)
        x = F.relu(self.fc1(x.view(-1, x.data.size()[1] * x.data.size()[2])))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


##########


class NetGIT(nn.Module):
    def __init__(self, nb_hidden):
        super(NetGIT, self).__init__()
        self.conv1 = nn.Conv1d(28, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=2)
        self.fc1 = nn.Linear(128 * 3, nb_hidden * 3)
        self.fc2 = nn.Linear(nb_hidden * 3, nb_hidden * 2)
        self.fc3 = nn.Linear(nb_hidden * 2, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool1d(self.conv3(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, x.data.size()[1] * x.data.size()[2])))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


##########


class NetConv(nn.Module):
    def __init__(self, nb_hidden):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv1d(28, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 11, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=2, stride=2))
        nn.Dropout()
        x = F.relu(self.fc1(x.view(-1, x.data.size()[1] * x.data.size()[2])))
        nn.Dropout()
        x = self.fc3(x)
        return x


##########


class NetCONV2D3(nn.Module):
    def __init__(self, nb_hidden):
        super(NetCONV2D3, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(20, 20, kernel_size=(3, 3))
        self.fc1 = nn.Linear(20 * 5 * 11, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # Nx28*50
        x = x.unsqueeze(1)  # Nx1x28x50

        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2), 2))  # Nx20x13x24
        x = F.relu(F.max_pool2d(self.conv2(x), (3, 2), 2))  # Nx20x5x11
        nn.Dropout()
        x = F.relu(self.fc1(x.view(-1, x.data.size()[1] * x.data.size()[2] * x.data.size()[3])))
        nn.Dropout()
        x = self.fc3(x)
        return x


########################################################################################################################
# Testing function


print("Beginning the training of the network 'Net_lin_dropout' \n\n\n")

# Select you model here!!!
# ||||||||||||||||||||||||
# VVVVVVVVVVVVVVVVVVVVVVVV

model = Net_lin_dropout(150)

nb_epochs = 100
batch_size = 50

lr = 1e-4

#################################

# CHANGE THE NUMBER OF TRAINING MODEL UNER HERE.
# IF YOU DO NOT WANT TO WAIT 300 YEARS FOR THE PROGRAM TO FINISH
nb_training = 20

train_and_test_model(model, nb_epochs, batch_size, lr, nb_training)

########################################################################################################################
print("Beginning the training of the network 'Net_lin' \n\n\n")

# Select you model here!!!
# ||||||||||||||||||||||||
# VVVVVVVVVVVVVVVVVVVVVVVV

model = Net_lin(150)

nb_epochs = 100
batch_size = 50

lr = 1e-4

#################################

# CHANGE THE NUMBER OF TRAINING MODEL UNER HERE.
# IF YOU DO NOT WANT TO WAIT 300 YEARS FOR THE PROGRAM TO FINISH
nb_training = 20

#train_and_test_model(model, nb_epochs, batch_size, lr, nb_training)

#######################################################################################################################
print("Beginning the training of the network 'NetConv' \n\n\n")

# Select you model here!!!
# ||||||||||||||||||||||||
# VVVVVVVVVVVVVVVVVVVVVVVV

model = NetConv(50)

nb_epochs = 190
batch_size = 75

lr = 1e-4


#################################

# CHANGE THE NUMBER OF TRAINING MODEL UNER HERE.
# IF YOU DO NOT WANT TO WAIT 300 YEARS FOR THE PROGRAM TO FINISH
nb_training = 20

#train_and_test_model(model, nb_epochs, batch_size, lr, nb_training)
########################################################################################################################
print("Beginning the training of the network 'NetDrop' \n\n\n")

# Select you model here!!!
# ||||||||||||||||||||||||
# VVVVVVVVVVVVVVVVVVVVVVVV

model = NetDROP(50)

nb_epochs = 130
batch_size = 75

lr = 1e-4

#################################

# CHANGE THE NUMBER OF TRAINING MODEL UNER HERE.
# IF YOU DO NOT WANT TO WAIT 300 YEARS FOR THE PROGRAM TO FINISH
nb_training = 20

train_and_test_model(model, nb_epochs, batch_size, lr, nb_training)
########################################################################################################################
print("Beginning the training of the network 'NetCONV2D3' \n\n\n")

model = NetCONV2D3(50)

nb_epochs = 100
batch_size = 50

lr = 5e-4

#################################

# CHANGE THE NUMBER OF TRAINING MODEL UNER HERE.
# IF YOU DO NOT WANT TO WAIT 300 YEARS FOR THE PROGRAM TO FINISH
nb_training = 20

train_and_test_model(model, nb_epochs, batch_size, lr, nb_training)
