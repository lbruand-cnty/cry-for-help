import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import datetime
import re
from random import shuffle
import math
from torch.utils.data import Dataset, DataLoader

class MLApi:
    def __init__(self, feature_index={}, labels_index={"GRAIT": 0, "GRAMC": 1}, epochs=10, select_per_epoch=200, num_labels=2, vocab_size=0):
        self.feature_index = feature_index
        self.labels_index = labels_index
        self.epochs = epochs # number of epochs per training session
        self.select_per_epoch = select_per_epoch  # number to select per epoch per label
        self.num_labels = num_labels
        self.vocab_size = vocab_size


    def create_features(self, df_unlabeled, df_training_data, minword=3):
        """Create indexes for one-hot encoding of words in files

        """
        total_training_words = {}
        for item in df_unlabeled + df_training_data:
            text = item
            for word in text.split():
                if word not in total_training_words:
                    total_training_words[word] = 1
                else:
                    total_training_words[word] += 1

        feature_index = {}
        for item in df_unlabeled + df_training_data:
            text = item
            for word in text.split():
                if word not in feature_index and total_training_words[word] >= minword:
                    feature_index[word] = len(feature_index)
        self.feature_index = feature_index
        return feature_index

    def make_feature_vector(self, features, feature_index):
        vec = torch.zeros(len(feature_index))
        for feature in features:
            if feature in feature_index:
                vec[feature_index[feature]] += 1
        return vec.view(1, -1)



    def train_model(self, training_data, test_data=None):
        """Train model on the given training_data

        Tune with the validation_data
        Evaluate accuracy with the evaluation_data
        """

        Xtrain, Ytrain = self.build_tensors(training_data)
        Xtest, Ytest = self.build_tensors(test_data)



        data_set = Data(Xtrain, Ytrain)
        trainloader = DataLoader(dataset=data_set, batch_size=64)
        print(data_set.y.shape)
        criterion = nn.CrossEntropyLoss()
        self.model = Net(Xtrain.shape[-1], 32, Ytrain.shape[-1])

        learning_rate = 0.01
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        n_epochs = 10
        train_loss_list = []
        test_loss_list = []

        # n_epochs
        for epoch in range(n_epochs):

            for x, y in trainloader:
                # clear gradient
                optimizer.zero_grad()
                # make a prediction
                z = self.model(x)

                train_loss = criterion(z, y)
                # calculate gradients of parameters
                train_loss.backward()
                # update parameters
                optimizer.step()

                train_loss_list.append(train_loss.data)
                with torch.no_grad():
                    y_test_hat = self.model(Xtest)
                    test_loss = criterion(y_test_hat, Ytest)
                    test_loss_list.append(test_loss)

                print(f"epoch {epoch}, train train_loss : {train_loss.item()}, test train_loss : {test_loss}")

    def build_tensors(self, data):
        inputs = [
            (self.feature_index, data.texts, " "),
            (self.labels_index, data.label, ",")
        ]
        return tuple(list([self.remove_middle_dim(self.extract_tensor_from_series(*input)) for input in inputs]))

    def extract_tensor_from_series(self, index, series, split_char) -> torch.Tensor:
        return torch.stack([self.make_feature_vector(text.split(split_char), index) for text in series.to_list()])

    def remove_middle_dim(self, x):
        return torch.reshape(x, (x.shape[0], x.shape[-1]))


class Data(Dataset):
    def __init__(self, x_train, y_train):
        self.x=x_train
        self.y=y_train
        self.len=self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


def train_model(df_train, df_test):
    print("train model")