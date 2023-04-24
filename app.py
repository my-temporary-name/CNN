import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as trans

import itertools

import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st

print('PyTorch Version: ', torch.__version__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# build a neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        
        return x

# calculate the number of correct predictions given the labels 
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

train_set = torchvision.datasets.FashionMNIST(
    root = './dataset/FashionMNIST',
    train = True,
    download = True,
    transform=trans.Compose([
        trans.ToTensor()
    ])
)

dataloader = torch.utils.data.DataLoader(train_set, batch_size=100)

model = CNN()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
EPOCH = 10


losses = []
corrects = []

for epoch in range(EPOCH):

    total_loss = 0
    total_correct = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
    
        preds = model(images) # make predictions
        loss = loss_fn(preds, labels) # calculate loss
    
        optimizer.zero_grad()
        loss.backward() # calculate gradients
        optimizer.step() # update weights
    
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    
    losses.append(total_loss)
    corrects.append(total_correct)
    
    print('EPOCH ', epoch+1,
          '\n------------------'
          '\nloss:', total_loss, 
          '\nnum of correct predictions: ', total_correct, '/', len(train_set))
    print()

# Plotting the graph
epochs = list(range(1, EPOCH+1))

# plt.plot(epochs, losses)
# plt.title("Epoch vs Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

fig1, ax = plt.subplots()
ax.plot(epochs, losses)
ax.set_title("Epoch vs Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
fig1.tight_layout()
# plt.show()
# st.pyplot(fig1)

# plt.plot(epochs, corrects)
# plt.title("Epoch vs Total Correct Predictions")
# plt.xlabel("Epoch")
# plt.ylabel("Total Correct Predictions")
# plt.show()

fig2, ax = plt.subplots()
ax.plot(epochs, corrects)
ax.set_title("Epoch vs Total Correct Predictions")
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Correct Predictions")
fig2.tight_layout()
# plt.show()

# get all the predictions for the entire training set
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    model.eval() # set model to evaluate mode
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    
    return all_preds 

with torch.no_grad(): # disable gradient computations 
    train_preds = get_all_preds(model, dataloader)
    # print(train_preds.shape) # (6000,10) 
    # print(train_preds.requires_grad) # False

print('true labels: ', train_set.targets.numpy())
print('pred labels: ', train_preds.argmax(dim=1).numpy())


# # we stacks the true labels and the predicted label, so that the first 
# # col is the true label, and the second col is the predicted label
# stacked = torch.stack((train_set.targets, train_preds.argmax(dim=1)), dim=1)
# print(stacked.numpy()) 


# # initialize a confusion matrix with 0s
# confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)
# print(confusion_matrix.numpy())



# # fill in the matrix
# for row in stacked:
#     true_label, pred_label = row.numpy()
#     confusion_matrix[true_label, pred_label] += 1

# print(confusion_matrix.numpy())



# simply call the confusion_matrix function to build a confusion matrix
cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
print(cm) 


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):

     
    if cmap is None:
        cmap = plt.get_cmap('Oranges')
    # fig, ax = plt.subplots()
    # plt.figure(figsize=(8, 6))
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(ax.imshow(cm, interpolation='nearest', cmap=cmap))
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    # plt.tight_layout()
    # plt.ylim(len(target_names)-0.5, -0.5)
    # plt.ylabel('True labels')
    # plt.xlabel('Predicted labels')
    # plt.savefig(title + '.png', dpi=500, bbox_inches = 'tight')
    # plt.show()
    
    # fig,ax = plt.subplots()
    # ax.tight_layout()
    # ax.ylim(len(target_names)-0.5, -0.5)
    # ax.ylabel('True labels')
    # ax.xlabel('Predicted labels')
    # plt.show()


    ax.set_ylim(len(target_names)-0.5, -0.5)
    ax.set_ylabel('True labels')
    ax.set_xlabel('Predicted labels')
    fig.tight_layout()
    fig.savefig(title + '.png', dpi=500, bbox_inches='tight')
    # plt.show()

    return fig 
# a tuple for all the class names
target_names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
figure = plot_confusion_matrix(cm, target_names)



st.title("Confusion Matrix for MNIST")
st.write("This app gives vizualization of the confusion matrix from scratch. Input: y_pred and y_hat")
st.pyplot(figure)

st.write("Plot of Epoch vs Loss")
st.pyplot(fig1)

st.write("Plot of Epoch vs Total correct prediction")
st.pyplot(fig2)

