import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import Module
from torch.utils.data import Dataset

from sklearn.utils import shuffle
import numpy as np

import numpy as np
import cv2
from glob import glob

import sys


class PneumoniaImages(Dataset):
    def __init__(self, train: bool, image_size: int = 150, add_contrast = False, small=False):
        super(PneumoniaImages, self).__init__()

        path = '../../Datasets/chest_xray/'
        if small:
            path = '../../Datasets/chest_xray/small/%i/' %(image_size)
        path += 'train/' if train else 'test/'
        path_normal = path + 'NORMAL/*'
        path_pneumonia = path + 'PNEUMONIA/*'

        normal_files = glob(path_normal)
        pneumonia_files = glob(path_pneumonia)

        X = np.zeros((len(normal_files) + len(pneumonia_files), 3,
                image_size, image_size), dtype=np.float32)

        Y = np.array([0] * len(normal_files) + [1] * len(pneumonia_files), dtype=np.float32).reshape((-1, 1))

        print('Starting with normal images')
        for i in range(len(normal_files)):
            img = cv2.imread(normal_files[i])
            if not small:
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

            if add_contrast:
                img = PneumoniaImages.__add_contrast(img, 127)

            img = img.transpose(2, 0, 1)
            img = np.array(img / 255, dtype=np.float32)

            X[i] = img

        print('Starting with pneumonia images')
        for i in range(len(pneumonia_files)):
            img = cv2.imread(pneumonia_files[i])
            if not small:
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

            if add_contrast:
                img = PneumoniaImages.__add_contrast(img, 127)
                
            img = img.transpose(2, 0, 1)
            img = np.array(img / 255, dtype=np.float32)

            X[i + len(normal_files)] = img

        #if add_contrast:
        #    X = PneumoniaImages.__add_contrast(X, 127)

        X, Y = shuffle(X, Y)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

    @staticmethod
    def __add_contrast(images, contrast):
        images = np.int16(images)
        images = images * (contrast / 127 + 1) - contrast
        images = np.clip(images, 0, 255)
        return np.uint8(images)



class VGG16(Module):
    def __init__(self, image_size: int = 150, cuda: bool = True):
        super(VGG16, self).__init__()
        self.image_size = image_size
        self.is_cuda = cuda
        self.training = False

        self.padding = nn.modules.ZeroPad2d(1)

        # Convolutional layers
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))

        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))

        self.conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3))
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))

        self.conv51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.conv52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.conv53 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))

        # Feedforward layers
        self.ff1 = nn.Linear(in_features=512 * ((image_size // (2 ** 5)) ** 2), out_features=4096)
        self.out = nn.Linear(in_features=4096, out_features=1)

    def forward(self, X):
        if self.is_cuda and not X.is_cuda:
            X = X.cuda()

        DROP_PROB = 0.5

        # First super layer
        # First layer
        X = self.padding(X)
        X = self.conv11(X)
        X = self.conv12(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Second super layer
        # Third layer
        X = self.padding(X)
        X = self.conv21(X)
        X = F.relu(X)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Fourth layer
        X = self.padding(X)
        X = self.conv22(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Third super layer
        # Fifth layer
        X = self.padding(X)
        X = self.conv31(X)
        X = F.relu(X)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Sixth layer
        X = self.padding(X)
        X = self.conv32(X)
        X = F.relu(X)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)
        
        # Seventh layer
        X = self.padding(X)
        X = self.conv33(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Fourth super layer
        # Eighth layer
        X = self.padding(X)
        X = self.conv41(X)
        X = F.relu(X)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Ninth layer
        X = self.padding(X)
        X = self.conv42(X)
        X = F.relu(X)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Tenth layer
        X = self.padding(X)
        X = self.conv43(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Fifth super layer
        # Eleventh layer
        X = self.padding(X)
        X = self.conv51(X)
        X = F.relu(X)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Twelveth layer
        X = self.padding(X)
        X = self.conv52(X)
        X = F.relu(X)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Thirteenth layer
        X = self.padding(X)
        X = self.conv53(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.dropout2d(X, p=DROP_PROB, training=self.training)

        # Fourteenth layer
        # Flatten
        X = X.reshape(-1, 512 * ((self.image_size // (2 ** 5)) ** 2))

        # Fifteenth layer
        X = self.ff1(X)
        X = F.relu(X)
        X = F.dropout(X, p=DROP_PROB, training=self.training)

        # Sixteenth layer
        return torch.sigmoid(self.out(X))

    def fit(self, train, test, lr=0.001, betas=(0.9, 0.999), epochs=10):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)

        train_costs = []
        test_costs = []
        train_accuracies = []
        test_accuracies = []
        self.training = True
        total = 0

        for epoch in range(epochs):
            epoch_cost = 0
            epoch_accuracy = 0
            for batch in train:
                images, labels = batch
                labels = labels.cuda()
                images = images.cuda()

                p = self(images)
                loss = F.binary_cross_entropy(p, labels)
                optimizer.zero_grad()
                loss.backward()
                epoch_cost += loss.item() / len(batch)
                optimizer.step()

                epoch_accuracy += self.__get_num_corrects(p, labels)
                total += len(batch)

            print(epoch_accuracy, total)
            epoch_accuracy /= total
            train_costs.append(epoch_cost)
            train_accuracies.append(epoch_accuracy)

            # Testing
            test_loss, test_accuracy = self.__get_loss(test)
            test_costs.append(test_loss)
            test_accuracies.append(test_accuracy)

            message = 'Epoch: %i, Train cost: %.3f, ' % (epoch+1, epoch_cost)
            message += 'Train acc: %.3f, Test cost: %.3f, ' % (epoch_accuracy, test_loss)
            message += 'Test acc: %.3f' % (test_accuracy)

            print(message)
        
        self.training = False

        return train_costs, train_accuracies, test_costs, test_accuracies


    def __get_loss(self, X):
        cost = 0
        accuracy = 0
        for batch in X:
            images, labels = batch
            labels = labels.cuda()
            p = self(images)

            loss = F.binary_cross_entropy(p, labels) / len(batch)
            cost += loss.item()
            accuracy += self.__get_num_corrects(p, labels)

        return cost, accuracy / len(X)

    @torch.no_grad()
    def predict(self, X):
        all_p = []
        for batch in X:
            images, _ = batch
            p = self(images)
            p = F.softmax(p, dim=1)

            all_p += p

        return torch.tensor(all_p)

    def __get_num_corrects(self, preds, labels):
        preds = torch.round(preds)
        return torch.sum((preds == labels).to(torch.float32), dim=0).item()
