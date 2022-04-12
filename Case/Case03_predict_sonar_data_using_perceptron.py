"""用perceptron预测声呐探测数据
数据集信息：
使用“sonar all data”，内含sonar.mines和sonar.rocks的208个样本数据。
文件“sonar.mines”包含 111 种模式，这些模式是通过在各种角度和各种条件下从金属圆柱体上反射声纳信号而获得的。
文件“sonar.rocks”包含在类似条件下从岩石中获得的 97 种模式。发射的声纳信号是频率调制的啁啾，频率上升。
该数据集包含从各种不同角度获得的信号，圆柱体跨越 90 度，岩石跨越 180 度。
每个模式都是 0.0 到 1.0 范围内的 60 个数字的集合。每个数字代表特定频带内的能量，在特定时间段内积分。
较高频率的积分孔径出现在较晚的时间，因为这些频率在啁啾期间传输较晚。
如果对象是岩石，则与每条记录关联的标签包含字母“R”，如果是矿井（金属圆柱体），则包含字母“M”。标签中的数字按角度的递增顺序排列，但它们不直接编码角度。
"""

# 1. import lib
from random import seed
from random import randrange
from csv import reader


# 2.write a csv/data reader
def read_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# 3.change string datatype
def change_string_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


filename = 'C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/sonar_all_data.csv'
dataset = read_csv(filename)
counter = 0.0
for i in range(len(dataset[0]) - 1):  # 一共61列，前60列都是属性，第61列是结果y。由counter可知，这里只转换前60列为浮点型。
    change_string_to_float(dataset, i)
    counter += 1
print(dataset)
print(counter)


# 4.change string column(class) to int
# 重点关注，核心思想在于通过i进行每一列的数据类型转换（i）

def change_str_column_to_int(dataset, column):
    class_value = [row[column] for row in dataset]
    unique_value = set(class_value)

    search_tool = dict()

    for i, value in enumerate(unique_value):
        search_tool[value] = i

    for row in dataset:
        row[column] = search_tool[row[column]]
    return search_tool


# 5. using k_folds cross validation

def k_folds_cross_validation(dataset, n_folds):
    dataset_split = list()
    # 对数据进行操作的时候，最好不要损坏原数据
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# 6.calculate the accuracy of our model
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# 7. whether the algo is good or not ?

def whether_the_algo_is_good_or_not(dataset, algo, n_folds, *args):
    folds = k_folds_cross_validation(dataset, n_folds)
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
        accuracy = calculate_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# 在第4步change string column(class) to int时，讲class分成0和1两类，下面将依据此分类计算权重参数weights并应用激活函数返回0或1.
# 8.make prediction

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0


# 9.using sgd(stochastic gradient descent) method to estimate weights

def estimate_our_weight_using_sgd_method(training_data, learning_rate, n_epoch):
    weights = [0.0 for i in range(len(training_data[0]))]
    for epoch in range(n_epoch):
        for row in training_data:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
    return weights #这里通过n_epoch次迭代（梯度下降），计算出一组参数weights


# 10. using sgd method to make perceptron algo's prediction

def perceptron(training_data, testing_data, learning_rate, n_epoch):
    predictions = list()
    weights = estimate_our_weight_using_sgd_method(training_data, learning_rate, n_epoch)
    for row in testing_data:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions


# 11.using real sonar dataset

seed(1)

filename = 'C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/sonar_all_data.csv'
dataset = read_csv(filename)
for i in range(len(dataset[0]) - 1):
    change_string_to_float(dataset, i)

change_str_column_to_int(dataset, len(dataset[0]) - 1)

n_folds = 3
learning_rate = 0.01
n_epoch = 500
scores = whether_the_algo_is_good_or_not(dataset, perceptron, n_folds, learning_rate, n_epoch)

print("The score of our model is : %s " % scores)
print("The average accuracy is : %3.f%% ， The baseline is 50%%" % (sum(scores) / float(len(scores))))

"""scores的计算过程：
整个数据分n个fold，在n中每次选一个fold作为测试集，其余fold作为训练集，遍历n次；
    在每一次fold中，在测试集fold在里计算perceptron算法，反馈测试集对应的预测值（即预测结果）；
        perceptron算法中，核心是使用激活函数反馈0或1（结果是二分情况），而激活函数的参数需要不断试出来；
            激活函数的参数weights是通过随机梯度下降方法，经过n_epoch次试错，返回一组参数，并应用于perceptron的激活函数中。
    在每一次fold中，将预测值与实际值对比，可返回：在当前fold内perceptron算法经过训练集学习后计算测试集的准确度。
n个fold，可以返回n个准确度，取平均值，可反映perceptron算法的预测准确情况。
"""