import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
from sklearn.metrics import roc_auc_score

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
    
    def train(self, inputs, labels, learning_rate=0.1, epochs=100):
        # Loss function is binary crossentropy
        loss_function = nn.BCELoss()
        # Optimizer is stochastic gradient descent
        SGD = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            pass

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

    # Using car dataset
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
    income_test_dataset['label'] = ''
        # Create copy of training dataset for predicting
    income_predicted_train_dataset = pd.DataFrame(income_train_dataset)
    income_predicted_train_dataset['label'] = ''   # or = np.nan for numerical columns

    data = Data()



    # Create csv file
    csv_predicton = 'prediction_decision_tree.csv'
    with open(csv_predicton, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(['ID', 'Prediction'])

        for i, _ in enumerate(y_test_predicted, start=1):
            csv_writer.writerow([i, y_test_predicted[i-1]])

if __name__ == "__main__":
    main()