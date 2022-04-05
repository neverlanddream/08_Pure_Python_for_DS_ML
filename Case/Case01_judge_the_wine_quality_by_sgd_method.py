"""Case01: Wine Quality Judgement"""
"""思路
1.load CSV
2.convert string to float
3.normalization
4.cross validation
5.evaluate our algo(RMSE)
"""


# 1. Import standard Lib

from csv import reader
from math import sqrt
from random import randrange
from random import seed


# 2.Load our csv file

def csv_loader(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# 文件存放在Base_Data下。代码中的路径用向左的斜杠。
# dataset_list = csv_loader('C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/winequality-white.csv')
# print(dataset_list)


# 3.Convert our datatype

def string_to_float_converter(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# 4.find the min and max of our dataset
def find_the_min_and_max_of_our_dataset(dataset):
    min_max_list = list()
    for i in range(len(dataset[0])):
        col_value = [row[i] for row in dataset]
        max_value = max(col_value)
        min_value = min(col_value)
        min_max_list.append([min_value, max_value])
    return min_max_list


# 5.normalize our data

def normalization(dataset, min_max_list):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max_list[i][0]) / (min_max_list[i][1] - min_max_list[i][0])


# 6.split our data

def k_fold_cross_validation_split(dataset, n_folds):
    splitted_dataset = list()
    copy_dataset = list(dataset)
    every_fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < every_fold_size:
            index = randrange(len(copy_dataset))
            fold.append(copy_dataset.pop(index))
        splitted_dataset.append(fold)
    return splitted_dataset


# 7.using root mean squared error method to calculate our model

def rmse_method(actual_data, predicted_data):
    sum_of_error = 0.0
    for i in range(len(actual_data)):
        predicted_error = predicted_data[i] - actual_data[i]
        sum_of_error += (predicted_error ** 2)
    mean_error = sum_of_error / float(len(actual_data))
    rmse = sqrt(mean_error)
    return rmse


# 8.how good is our algo by using cross validation

def how_good_is_our_algo(dataset, algo, n_folds, *args):
    folds = k_fold_cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algo(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = rmse_method(actual, predicted)
        scores.append(rmse)
    return scores


# 9.make prediction

def predict(row, coefficients):
    y_hat = coefficients[0]
    for i in range(len(row)-1):
        y_hat += coefficients[i+1] * row[i]
    return y_hat


# 10.using stochastic gradient descent method to calculate the coeffient

def sgd_method_to_calculate_coefficient(training_data, learning_rate, n_epoch):
    coeffient_list = [0.0 for i in range(len(training_data[0]))]
    for epoch in range(n_epoch):
        for row in training_data:
            y_hat = predict(row, coeffient_list)
            error = y_hat - row[-1]
            coeffient_list[0] = coeffient_list[0] - learning_rate * error
            for i in range(len(row)-1):
                coeffient_list[i+1] = coeffient_list[i+1] - learning_rate * error *row[i]
            # print(learning_rate, n_epoch, error)
    return  coeffient_list


# 11.using linear regresion algo

def using_sgd_method_to_calculate_linear_regression(training_data, testing_data, learning_rate, n_epoch):
    predictions = list()
    coeffient_list = sgd_method_to_calculate_coefficient(training_data, learning_rate, n_epoch)
    for row in testing_data:
        y_hat = predict(row, coeffient_list)
        predictions.append(y_hat)
    return predictions


# 12.Using our real wine quality data
seed(1)
wine_quality_data_name = 'C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/winequality-white.csv'
dataset = csv_loader(wine_quality_data_name)
for i in range(len(dataset[0])):
    string_to_float_converter(dataset,i)


# 13.Normalization

min_and_max = find_the_min_and_max_of_our_dataset(dataset)
normalization(dataset, min_and_max)


# 14.How good is our algo
n_fold = 10
learning_rate = 0.01
n_epoch = 50

algo_score = how_good_is_our_algo(dataset, using_sgd_method_to_calculate_linear_regression, n_fold, learning_rate, n_epoch)

print("Our algo's score is %s" % algo_score)
print("The mean of our algo's RMSE is %.3f" % (sum(algo_score)/float(len(algo_score))))