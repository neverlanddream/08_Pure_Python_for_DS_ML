# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)
"""

"""


from math import sqrt
from math import pi
from math import exp

# 1 将dataset依据结果值分类，并将各样本依分类放入字典，最后返回这个字典。
def split_our_data_by_class(dataset):
    splited_data = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in splited_data):
            splited_data[class_value] = list()
        splited_data[class_value].append(vector)
    return splited_data #返回字典{分类：相对应分类的样本数据}

# 2
def calculate_the_mean(a_list_of_num):
    mean = sum(a_list_of_num) / float(len(a_list_of_num))
    return mean

# 3
def calculate_the_standard_deviation(a_list_of_num):
    the_mean = calculate_the_mean(a_list_of_num)
    the_variance = sum([(x - the_mean) ** 2 for x in a_list_of_num]) / float(len(a_list_of_num) - 1)
    std = sqrt(the_variance)
    return std

# 4
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
# 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
# *在zip()中是做解包操作，详见链接说明：
# https://blog.csdn.net/PaulZhn/article/details/104391756
def describe_our_data(dataset):
    description = [(calculate_the_mean(column),
                    calculate_the_standard_deviation(column),
                    len(column)) for column in zip(*dataset)] #相当于将dataset转置为三个元组，对这三个元组分别计算mean、std、len
    # print(list(zip(*dataset)))
    del (description[-1]) # 将对结果column的description删除，保留对各样本属性culumn的description
    # print(description)
    return description # 返回各样本属性的mean、std、count。[(mean, std, count),(),...]

# 5
# Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组。
def describe_our_data_by_class(dataset):
    splited_data = split_our_data_by_class(dataset)
    data_description = dict()
    for class_value, rows in splited_data.items():
        data_description[class_value] = describe_our_data(rows) # 分别对类别0和1对应的样本数据计算mean和std并赋值给字典对应键值
    # print(data_description)
    return data_description # 返回字典{分类：[(mean, std, count),(),...] ,分类：[], ... }

# 6
def calculate_the_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    result = (1 / (sqrt(2 * pi) * stdev)) * exponent
    return result

# 7
def calculate_class_probability(description, row):
    total_rows = sum([description[label][0][2] for label in description])
    probabilities = dict()
    for class_value, class_description in description.items():
        probabilities[class_value] = description[class_value][0][2] / float(total_rows)
        for i in range(len(class_description)):
            mean, stdev, count = class_description[i]
            probabilities[class_value] *= calculate_the_probability(row[i], mean, stdev)
    return probabilities


dataset = [[0.8, 2.3, 0],
           [2.1, 1.6, 0],
           [2.0, 3.6, 0],
           [3.1, 2.5, 0],
           [3.8, 4.7, 0],
           [6.1, 4.4, 1],
           [8.6, 0.3, 1],
           [7.9, 5.3, 1],
           [9.1, 2.5, 1],
           [6.8, 2.7, 1]]

description = describe_our_data_by_class(dataset)
# print(description)
probability = calculate_class_probability(description, dataset[0])
print(probability)
