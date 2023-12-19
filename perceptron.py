import os
import numpy as np
import pandas as pd
from copy import deepcopy
import csv
from sklearn.metrics import roc_auc_score

class Perceptron:
    def train_standard(self, train_dataset, epochs, learning_rate):
        # Initialize weights. Bias is the first element
        weights = [0.0 for i in range(len(train_dataset.columns))]
        for epoch in range(epochs):
            # Shuffle the data
            train_dataset = train_dataset.sample(frac=1.0)
            for index, dataset_row in train_dataset.iterrows():
                row = dataset_row.tolist()
                prediction = self._predict_row(row, weights)
                # Set the correct label for calculation
                if row[-1] == 0.0:
                    actual_label = -1.0
                else:
                    actual_label = 1.0
                if prediction != actual_label:
                    weights[0] += learning_rate * actual_label
                    for i in range(len(row) - 1):
                        # Update weights
                        weights[i+1] = weights[i+1] + learning_rate * actual_label * row[i]
        return weights
    
    def train_voted(self, train_dataset, epochs, learning_rate):
        # Initialize weights. Bias is the first element
        weights = [0.0 for i in range(len(train_dataset.columns))]
        weight_vectors = []
        votes = []
        vote_count = 0
        for epoch in range(epochs):
            # Shuffle the data
            train_dataset = train_dataset.sample(frac=1.0)
            for index, dataset_row in train_dataset.iterrows():
                row = dataset_row.tolist()
                prediction = self._predict_row(row, weights)
                # Set the correct label for calculation
                if row[-1] == 0.0:
                    actual_label = -1.0
                else:
                    actual_label = 1.0
                if prediction != actual_label:
                    weights[0] += learning_rate * actual_label
                    for i in range(len(row) - 1):
                        # Add weight vector and its vote
                        weight_vectors.append(deepcopy(weights))
                        votes.append(vote_count)
                        # Create new weight vector
                        weights[i+1] = weights[i+1] + learning_rate * actual_label * row[i]
                        vote_count = 1
                else:
                    vote_count += 1
        return weight_vectors, votes
    
    def train_averaged(self, train_dataset, epochs, learning_rate):
        # Initialize weights. Bias is the first element
        weights = [0.0 for i in range(len(train_dataset.columns))]
        averages = [0.0 for i in range(len(train_dataset.columns))]
        for epoch in range(epochs):
            # Shuffle the data
            train_dataset = train_dataset.sample(frac=1.0)
            for index, dataset_row in train_dataset.iterrows():
                row = dataset_row.tolist()
                prediction = self._predict_row(row, weights)
                # Set the correct label for calculation
                if row[-1] == 0.0:
                    actual_label = -1.0
                else:
                    actual_label = 1.0
                if prediction != actual_label:
                    weights[0] += learning_rate * actual_label
                    for i in range(len(row) - 1):
                        # Update weights
                        weights[i+1] = weights[i+1] + learning_rate * actual_label * row[i]
                else:
                    for i in range(len(weights)):
                        averages[i] += weights[i]
        return averages

    def _predict_row(self, row, weights):
        # The bias is the first element of weights vector
        activation = weights[0]
        for i in range(len(row) - 1):
            # Compute the dot product for the rest of the elements
            activation = activation + weights[i+1] * row[i]
        if activation >= 0.0:
            return 1
        else:
            return -1
        
    def predict_standard(self, dataset, weights):
        for index, dataset_row in dataset.iterrows():
            row = dataset_row.tolist()
            prediction = self._predict_row(row, weights)
            if prediction >= 0.0:
                dataset.at[index, 'label'] = 1
            else:
                dataset.at[index, 'label'] = 0
        return dataset

    def predict_voted(self, dataset, weights, votes):
        for index, dataset_row in dataset.iterrows():
            row = dataset_row.tolist()
            voted_prediction = 0
            for i in range(len(weights)):
                prediction = 0
                for j in range(len(row) - 1):
                    prediction = prediction + weights[i][j+1] * row[j]
                if prediction >= 0.0:
                    prediction = 1
                else:
                    prediction = -1
                voted_prediction += votes[i] * prediction
            if voted_prediction >= 0.0:
                dataset.at[index, 'label'] = 1
            else:
                dataset.at[index, 'label'] = 0
        return dataset

    def predict_averaged(self, dataset, weights):
        for index, dataset_row in dataset.iterrows():
            row = dataset_row.tolist()
            prediction = self._predict_row(row, weights)
            if prediction >= 0.0:
                dataset.at[index, 'label'] = 1
            else:
                dataset.at[index, 'label'] = 0
        return dataset
    
    def compute_error(self, actual_labels, predicted_labels):
        incorrect_examples = 0
        for i in range(len(actual_labels)):
            if actual_labels[i] != predicted_labels[i]:
                incorrect_examples += 1
        return incorrect_examples / len(actual_labels)
        
def main():
    # Get the directory of the script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV files
    income_train_path = os.path.join(script_directory, 'Datasets', 'income', 'train.csv')
    income_test_path = os.path.join(script_directory, 'Datasets', 'income', 'test.csv')

    perceptron = Perceptron()

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
    income_test_dataset['label'] = 0

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

    # Create copy of training dataset for predicting
    income_predicted_train_dataset = pd.DataFrame(income_train_dataset)
    income_predicted_train_dataset['label'] = ''   # or = np.nan for numerical columns
    # Results using standard perceptron
    standard_weights = perceptron.train_standard(income_train_dataset, 10, 0.5)
    income_predicted_train_dataset = perceptron.predict_standard(income_predicted_train_dataset, standard_weights)
    income_test_dataset = perceptron.predict_standard(income_test_dataset, standard_weights)
    ## Results using voted perceptron
    # voted_weights, votes = perceptron.train_voted(income_train_dataset, 10, 0.5)
    # income_predicted_train_dataset = perceptron.predict_voted(income_predicted_train_dataset, voted_weights, votes)
    # income_test_dataset = perceptron.predict_voted(income_test_dataset, voted_weights, votes)
    ## Results using averaged perceptron
    # averaged_weights = perceptron.train_averaged(income_train_dataset, 10, 0.5)
    # income_predicted_train_dataset = perceptron.predict_averaged(income_predicted_train_dataset, averaged_weights)
    # income_test_dataset = perceptron.predict_averaged(income_test_dataset, averaged_weights)
    # Get array predictions
    y_train = income_train_dataset['label'].to_numpy()
    y_train_predicted = income_predicted_train_dataset['label'].to_numpy()
    y_test_predicted = income_test_dataset['label'].to_numpy()
    income_training_error = perceptron.compute_error(y_train, y_train_predicted)

    print('The training error is', income_training_error)
    print('Score is', roc_auc_score(y_train, y_train_predicted))

    # Create csv file
    csv_predicton = 'prediction_standard_perceptron.csv'
    with open(csv_predicton, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(['ID', 'Prediction'])

        for i, _ in enumerate(y_test_predicted, start=1):
            csv_writer.writerow([i, y_test_predicted[i-1]])

if __name__ == "__main__":
    main()