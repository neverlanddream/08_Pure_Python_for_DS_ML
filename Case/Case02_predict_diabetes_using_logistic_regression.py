"""用SGD计算logisitc regression系数形成模型，并测试模型好坏
核心步骤：
1.
2.
3.
"""

from csv import reader
from math import exp
from random import randrange
from random import seed


# 1.Load our csv data

def load_csv(data_file):
    # 通过一个list容器去装数据
    data_set = list()
    with open(data_file, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            # 判断非空row，并装入容器
            if not row:
                continue
            data_set.append(row)
    return data_set

# diabetes_csv = 'C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/diabetes.csv'
# print(load_csv(diabetes_csv))


# 2.转换数据成float类型

def string_converter(data_set, column):
    for row in data_set:
        # 把前后空格去掉
        row[column] = float(row[column].strip())
    return

# diabetes_csv = 'C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/diabetes.csv'
# ret_data = load_csv(diabetes_csv)
# for i in range(len(ret_data[0])):
#     string_converter(ret_data,i)
# print(ret_data)


# 3.Find the min and max value of our data（找到是data中每一列/维度的最小最大值，并返回所有维度的最小最大值的list）

def find_the_min_and_max_of_our_data(dataset):
    min_and_max_list = list()
    for i in range(len(dataset[0])):
        # 为了找出每一列（维度）的最小最大值，先把每一列数放进values_in_every_column
        values_in_every_column = [row[i] for row in dataset]
        the_min_value = min(values_in_every_column)
        the_max_value = max(values_in_every_column)
        min_and_max_list.append([the_min_value, the_max_value])
    return min_and_max_list

# diabetes_csv = 'C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/diabetes.csv'
# ret_data = load_csv(diabetes_csv)
# for i in range(len(ret_data[0])):
#     string_converter(ret_data,i)
# min_and_max_list = find_the_min_and_max_of_our_data(ret_data)
# print(min_and_max_list)


# 4.Rescale our data so it fits to range 0~1【数据对象：样本集】

def rescale_our_data(dataset, min_max_list):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max_list[i][0]) / (min_max_list[i][1] - min_max_list[i][0])
    return dataset


# 5.k fold train and test split【数据对象：样本集】【样本集结构中增加一层fold（N个）包含各独立样本】

def k_fold_cross_validation(dataset, how_many_folds_do_you_want):
    # 最终想要的splitted_dataset是内含N个fold，每个fold里是随机选取的元素
    splitted_dataset = list()
    # 对原数据进行处理的时候，尽量不要改变源数据（可以通过创建copy的方式对copy数据进行处理）
    copy_dataset = list(dataset)
    how_big_is_every_fold = int( len(dataset) / how_many_folds_do_you_want )
    # 创建一个空的盒子，然后逐一选取数据放入盒子中
    for i in range(how_many_folds_do_you_want):
        box_for_my_fold = list()
        while len(box_for_my_fold) < how_big_is_every_fold:
            some_random_index_in_the_fold = randrange(len(copy_dataset))
            # pop() 方法删除字典给定键 key 所对应的值，返回值为被删除的值。
            box_for_my_fold.append(copy_dataset.pop(some_random_index_in_the_fold))
        splitted_dataset.append(box_for_my_fold)
    return splitted_dataset


# 6.Calculate the accuracy of our model【数据对象：样本个体】

def calculate_the_accuracy_of_our_model(actual_data, predicted_data):
    counter_of_correct_prediction = 0
    for i in range(len(actual_data)):
        if actual_data[i] == predicted_data[i]:
            counter_of_correct_prediction += 1
    return counter_of_correct_prediction/float(len(actual_data) )* 100.0


# 7.how good is our algo【引用10和6和5】【数据对象：样本集】【重点关注数据结构的变化】

def how_good_is_our_algo(dataset, algo, how_many_fold_do_you_want, *args):
    # folds是k_fold_cross_validation返回的splitted_dataset【数据结构是：样本集整体--样本集fold（N个）--样本个体--属性元素】
    folds = k_fold_cross_validation(dataset, how_many_fold_do_you_want)
    scores = list()
    for fold in folds: #【数据对象：fold】
        training_data_set = list(folds)  #此时，training_data_set数据结构是：样本集整体--样本集fold（N个）--样本个体--属性元素
        training_data_set.remove(fold)
        training_data_set = sum(training_data_set, []) #将training_data_set整体数据结构降维，没有样本集fold这层级了。示例见99小技巧

        testing_data_set = list()        #testing_data_set即将构建的数据结构与降维后的training_data_set相同：样本集整体--样本个体--属性元素
        # 保险操作，去除真实数据，避免影响模型的学习结果
        for row in fold: #【数据对象：样本个体】
            row_copy = list(row)
            testing_data_set.append(row_copy)
            row_copy[-1] = None
        predicted = algo(training_data_set,testing_data_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_the_accuracy_of_our_model(actual,predicted)
        accuracy = round(accuracy,3)
        scores.append(accuracy)
    return scores


# 8.Make prediction by using the coef 其实就是计算yhat，但是coef未知 【数据对象：样本个体】

def prediction(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i+1] * row[i]
    return 1/(1.0 + exp(-yhat))


# 9.Using stochastic gradient decent to estimate our coef of logistic regression【引用8】【数据对象：样本集】

def estimate_coef_of_lr_using_sgd_method(training_data, learning_rate, how_many_epoch_do_you_want):
    coef = [0.0 for i in range(len(training_data[0]))]
    for epoch in range(how_many_epoch_do_you_want):
        for row in training_data:
            yhat = prediction(row, coef)
            error = yhat - row[-1]
            gradient = yhat * (1.0 - yhat)
            coef[0] = coef[0] - learning_rate * error * gradient
            for i in range(len(row)-1):
                coef[i+1] = coef[i+1] - learning_rate * error * gradient * row[i]
    return coef


# 10.Logistic Regression prediction in our func 其实是yhat合集【引用9和8】【数据对象：样本集】

def logistic_regression(training_data, testing_data, learning_rate, how_many_epoch_do_you_want):
    predictions = list()
    coef = estimate_coef_of_lr_using_sgd_method(training_data, learning_rate, how_many_epoch_do_you_want)
    for row in testing_data:
        yhat = prediction(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return predictions


# Using Kaggle data to test our model
#
seed(1)
how_many_folds_do_you_want = 10
learning_rate = 0.1
how_many_epoch_do_you_want = 100

diabetes_csv = 'C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/diabetes.csv'
dataset = load_csv(diabetes_csv)
for i in range(len(dataset[0])):
    string_converter(dataset,i)
min_and_max_list = find_the_min_and_max_of_our_data(dataset)
dataset = rescale_our_data(dataset, min_and_max_list)
scores = how_good_is_our_algo(dataset, logistic_regression, how_many_folds_do_you_want, learning_rate, how_many_epoch_do_you_want)

print("The scores of our model are %s" % scores)
print("The average accuracy of our model is %.3f" % (sum(scores)/float(len(scores))))

# print(len(dataset))
# # print(splitted_dataset)
# print(len(splitted_dataset))
# print(min_and_max_list)