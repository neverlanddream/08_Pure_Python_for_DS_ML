"""使用insurance数据做一个简单数据回归的案例"""

from csv import reader
from math import sqrt
from random import randrange
from random import seed

# 1.Load our csv data


def load_csv(data_file):
    data_set = list()
    with open(data_file, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data_set.append(row)
    return data_set

# insurance_csv = 'C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/insurance.csv'
# print(load_csv(insurance_csv))
# 发现数据均为字符串，我们接下来对字符串进行转化，最终要转化成float


# 2.转换数据成float类型

def string_converter(data_set, column):
    for row in data_set:
        row[column] = float(row[column].strip())

# 用.strip()的原因
# a = "        99.9  ".strip()
# print(a)


# 3.RMSE(root mean squared error)模型预测的准确性基本判断方法\衡量模型的标尺

def calculate_RMSE(actual_data, predicted_data):
    sum_error = 0.0
    for i in range(len(actual_data)):
        predicted_error = predicted_data[i] - actual_data[i]
        sum_error += predicted_error ** 2
    mean_squared_error = sum_error / float(len(actual_data))
    return  sqrt(mean_squared_error)


# 4.模型到底如何（train/test split)

# pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
def train_test_split(data_set, split):
    train = list()
    train_size = split * len(data_set)
    data_set_copy = list(data_set)
    while len(train) < train_size:
        index = randrange(len(data_set_copy))
        train.append(data_set_copy.pop(index))
    return train, data_set_copy

# 通过在训练集，测试集切分后，用RMSE进行衡量模型好坏
# 因为要导入一个algo进来，而algo的参数未知，所以预留*args，为algo的参数准备。
def how_good_is_algo(data_set, algo, split, *args):
    train, test = train_test_split(data_set, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    # 伪代码思想，先用algo统一代替具体算法
    predicted = algo(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = calculate_RMSE(actual, predicted)
    return rmse


# 5.algo。为了实现简单线性回归的小玩意

def mean(values):
    return sum(values) / float(len(values))

def convariance(x, the_mean_of_x, y, the_mean_of_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - the_mean_of_x)*(y[i] - the_mean_of_y)
    return covar

def variance(values, mean):
    return sum([(x - mean) ** 2 for x in values])

# y = b1*x + b0
def coefficients(data_set):
    x = [row[0] for row in data_set]
    y = [row[1] for row in data_set]
    the_mean_of_x, the_mean_of_y = mean(x), mean(y)
    b1 = convariance(x, the_mean_of_x, y, the_mean_of_y) / variance(x, the_mean_of_x)
    b0 = the_mean_of_y - b1 * the_mean_of_x
    return [b0, b1]

# 6.这里写简单线性回归的具体预测

def using_simple_linear_regression(train, test):
    # 套路：先弄一个空的容器出来，然后逐一处理放入
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        y_hat = b1 * row[0] +b0
        predictions.append(y_hat)
    return predictions


# 7.带入真实数据

# 可调项，避免“魔法数字”
seed(1)
split = 0.7

# 读取数据
data_file = 'C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/insurance.csv'
data_set = load_csv(data_file)
# 数据准备
for i in range(len(data_set[0])):
    string_converter(data_set,i)

# 定义的using_simple_linear_regression(train, test)使用的两个参数在how_good_is_algo函数中引用algo时均在函数内定义并计算产生。
rmse = how_good_is_algo(data_set, using_simple_linear_regression, split)

print("RMSE of our algo is : %.3f" % (rmse))
