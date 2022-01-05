import torch
import torch.nn as nn
import pandas as pd
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


    def create_features(self, df_unlabeled, df_train, minword=3):
        """Create indexes for one-hot encoding of words in files

        """
        total_training_words = {}
        for item in df_unlabeled + df_train:
            text = item
            for word in text.split():
                if word not in total_training_words:
                    total_training_words[word] = 1
                else:
                    total_training_words[word] += 1

        feature_index = {}
        for item in df_unlabeled + df_train:
            text = item
            for word in text.split():
                if word not in feature_index and total_training_words[word] >= minword:
                    feature_index[word] = len(feature_index)
        self.feature_index = feature_index
        self.vocab_size = len(feature_index)
        return feature_index

    def make_feature_vector(self, features, feature_index):
        vec = torch.zeros(len(feature_index))
        for feature in features:
            if feature in feature_index:
                vec[feature_index[feature]] += 1
        return vec.view(1, -1)

    def get_low_conf_unlabeled(self, unlabeled_data, number=80, limit=10000):
        confidences = []
        if limit == -1:  # we're predicting confidence on *everything* this will take a while
            print("Get confidences for unlabeled data (this might take a while)")
        else:
            # only apply the model to a limited number of items
            unlabeled_data = unlabeled_data.sample(frac=1).reset_index(drop=True)
            rest = unlabeled_data[limit:]
            unlabeled_data_limited = unlabeled_data[:limit]


        with torch.no_grad():
            for idx, item in unlabeled_data_limited.iterrows():
                #textid = item[0]
                #item[3] = "random_remaining"
                print(item)
                text = item.texts
                # TODO : This should be do over the unlabeled set in batching.
                feature_vector = self.make_feature_vector(text.split(), self.feature_index)
                log_probs = self.model(feature_vector)

                # get confidence that it is related
                prob_related = math.exp(log_probs.data.tolist()[0][1])  # TODO : This is not working really.

                if prob_related < 0.5:
                    confidence = 1 - prob_related
                else:
                    confidence = prob_related

                #item[3] = "low confidence"
                #item[4] = confidence
                confidences.append(confidence)

        unlabeled_data_limited["confidence"] = confidences
        unlabeled_data_limited.sort_values(by=["confidence"], inplace=True)
        unlabeled_data_limited = unlabeled_data_limited.drop(columns=["confidence"]) # TODO : Keep the confidence for future use.

        return unlabeled_data_limited, rest

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
        self.model = Net(Xtrain.shape[-1], 32, Ytrain.shape[-1])  # TODO : Work with probits. ?

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