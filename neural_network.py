import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import itertools
import csv

# DELETE LATER ---------------------------------------------------------
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# DELETE LATER ---------------------------------------------------------

class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs=14, n_hidden_neurons=20, n_outputs=1):
        super(NeuralNetwork, self).__init__()
        # Create layers
        self.layer1 = nn.Linear(n_inputs, n_hidden_neurons)
        # Initialize weights using He initialization
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity="relu")
        self.layer2 = nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.layer3 = nn.Linear(n_hidden_neurons, n_outputs)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.sigmoid(self.layer3(x))
        return x
    
    def train(self, dataset, learning_rate=0.1, epochs=100):
        # Loss function is binary crossentropy
        loss_function = nn.BCELoss()
        losses = []
        # Optimizer is stochastic gradient descent
        SGD = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            for X, y in dataset:
                # Set gradients to zero
                SGD.zero_grad()
                prediction = self.forward(X)
                loss = loss_function(prediction, y.unsqueeze(-1))
                losses.append(loss.item())
                loss.backward()
                SGD.step()
        print('Training complete')

    def predict(self, dataset):
        y_predicted = []
        with torch.no_grad():
            for X, y in dataset:
                outputs = self.forward(X)
                # Create numpy array with predictions
                predicted = np.where(outputs < 0.5, 0, 1)
                # Convert array to regular list
                predicted = list(itertools.chain(*predicted))
                y_predicted.append(predicted)
            return y_predicted

    def predict(self, dataset):
        y_predicted = []
        y_actual = []
        total = 0
        incorrect = 0
        with torch.no_grad():
            for X, y in dataset:
                outputs = self.forward(X)
                # Create numpy array with predictions
                predicted = np.where(outputs < 0.5, 0, 1)
                # Convert array to regular list
                predicted = list(itertools.chain(*predicted))
                y_predicted.append(predicted)
                y_actual.append(y)
                total += y.size(0)
                incorrect += (predicted != y.numpy()).sum().item()
                print('Error is', incorrect / total)
            return y_predicted
        
    # def predict(self, dataset):
    #     y_predicted = []
    #     with torch.no_grad():
    #         for X, y in dataset:
    #             outputs = self.forward(X)
    #             # Create numpy array with predictions
    #             predicted = np.where(outputs < 0.5, 0, 1)
    #             # Convert array to regular list
    #             predicted = list(itertools.chain(*predicted))
    #             y_predicted.append(predicted)
    #         return y_predicted
        
    # def compute_error(self, y_actual, y_predicted):
    #     incorrect = 0
    #     for i in range(len(y_actual)):
    #         if y_actual[i] != y_predicted[i]:
    #             incorrect += 1
    #     print('Error is', incorrect / len(y_actual))
    #     return incorrect / len(y_actual)

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.size = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.size
    
def main():
    # Get the directory of the script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV files
    income_train_path = os.path.join(script_directory, 'Datasets', 'income', 'train.csv')
    income_test_path = os.path.join(script_directory, 'Datasets', 'income', 'test.csv')

    network = NeuralNetwork()

    # Upload training dataset
    income_train_dataset = pd.read_csv(income_train_path, header=None)
    income_train_dataset.columns = ['age','workclass','fnlwgt','education','education-num','marital-status',
                                    'occupation','relationship','race','sex', 'capital-gain', 'capital-loss', 
                                    'hours-per-week', 'native-country', 'label']
    income_features = {'age': [], 
                    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                                  'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'], 
                    'fnlwgt': [], 
                    'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                                  'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                                  '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'], 
                    'education-num': [], 
                    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 
                                       'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'], 
                    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                                       'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                                       'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                                       'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                                       'Armed-Forces'],
                    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], 
                    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'], 
                    'sex': ['Female', 'Male'], 
                    'capital-gain': [], 
                    'capital-loss': [], 
                    'hours-per-week': [], 
                    'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 
                                       'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 
                                       'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
                                       'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 
                                       'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 
                                       'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 
                                       'Peru', 'Hong', 'Holand-Netherlands']}
    # Upload testing dataset
    income_test_dataset = pd.read_csv(income_test_path, header=None)
    income_test_dataset.columns = ['age','workclass','fnlwgt','education','education-num','marital-status',
                                    'occupation','relationship','race','sex', 'capital-gain', 'capital-loss', 
                                    'hours-per-week', 'native-country']
    income_test_dataset['label'] = 1

    # Handle missing values
    for feature in income_train_dataset.columns:
        most_common_value = income_train_dataset[feature].value_counts().index[0]
        if most_common_value == '?':
                most_common_value = income_train_dataset[feature].value_counts().index[1]
        income_train_dataset[feature] = income_train_dataset[feature].replace('?', most_common_value)
        income_test_dataset[feature] = income_test_dataset[feature].replace('?', most_common_value)

    # Convert non-numeric values to numbers
    for feature, feature_values in income_features.items():
        numeric_value = 1
        for feature_value in feature_values:
            income_train_dataset[feature] = income_train_dataset[feature].replace(feature_value, numeric_value)
            income_test_dataset[feature] = income_test_dataset[feature].replace(feature_value, numeric_value)
            numeric_value += 1

    # count_train = income_train_dataset.apply(lambda x: x.value_counts().get('?', 0)).sum() #3306
    # count_test = income_test_dataset.apply(lambda x: x.value_counts().get('?', 0)).sum() #3159
    
    # Create arrays based on dataframes
    X_train = income_train_dataset.drop('label', axis=1).to_numpy()
    X_test = income_test_dataset.drop('label', axis=1).to_numpy()
    y_train = income_train_dataset['label'].to_numpy()
    y_test = income_test_dataset['label'].to_numpy()

    # Create train dataset
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=25000, shuffle=True)

    # Create test dataset
    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=25000, shuffle=True)

    network.train(train_dataloader)
    network.predict(train_dataloader)
    y_predicted = network.predict(test_dataloader)
    print()

    # # DELETE LATER ---------------------------------------------------------

    # t_network = NeuralNetwork(2, 10, 1)

    # X, y = make_circles(n_samples = 10000, noise= 0.05, random_state=26)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)

    # train_data = Data(X_train, y_train)
    # train_dataloader = DataLoader(dataset=train_data, batch_size=200, shuffle=True)

    # test_data = Data(X_test, y_test)
    # test_dataloader = DataLoader(dataset=test_data, batch_size=200, shuffle=True)

    # t_network.train(train_dataloader)
    # t_network.predict(test_dataloader)

    # # DELETE LATER ---------------------------------------------------------


    # Create csv file
    csv_predicton = 'prediction_neural_network.csv'
    with open(csv_predicton, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(['ID', 'Prediction'])

        for i, _ in enumerate(y_predicted, start=1):
            csv_writer.writerow([i, y_predicted[i-1]])

if __name__ == "__main__":
    main()