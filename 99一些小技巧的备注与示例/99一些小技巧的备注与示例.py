"""一些特殊用法、技巧的备注"""


# 1.关于用sum对列表求和降维的用法
"""
sum_for_list_v1 = [[2 , 2 , 0],
                   [2 , 4 , 0],
                   [9 , 1 , 1],
                   [10, 4 , 1]]

# sum_for_list_v2 = sum( sum_for_list_v1 , [] )
# print(sum_for_list_v2)

#不和空列表求和会报错
# sum_for_list_v3 = sum( sum_for_list_v1 )
# print(sum_for_list_v3)
"""


# 2.将string型样本属性数据和string型样本结果数据转化成浮点型。
"""
def change_string_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def change_str_column_to_int(dataset, column):
    class_value = [row[column] for row in dataset]
    unique_value = set(class_value)

    search_tool = dict()

    for i, value in enumerate(unique_value):
        search_tool[value] = i
    for row in dataset:
        row[column] = search_tool[row[column]]
    return search_tool


# enumerate在字典上是枚举、列举的意思
# 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
# 关于enumerate的用法可参考 https://www.jb51.net/article/177995.htm

# 示例：
string_dataset = [['2' , '2' , 'rock'],
                  ['2' , '4' , 'rock'],
                  ['9' , '1' , 'mine'],
                  ['10', '4' , 'mine']]

counter = 0.0
# for i in range(len(string_dataset[0]) - 1):  # 一共61列，前60列都是属性，第61列是结果y。由counter可知，这里只转换前60列为浮点型。
#     change_string_to_float(string_dataset, i)
#     counter += 1
# print("将数据属性转换为浮点型，处理%.3f次后，如下："%(counter))
# print(string_dataset)
#
# search_tool = change_str_column_to_int(string_dataset, len(string_dataset[0]) - 1)
# print("将string型的数据结果转换为浮点型，并映射，处理后，映射关系和数据如下：")
# print(search_tool)
# print(string_dataset)
"""


# 3.查看非空，or
"""
left = []
right = []

# if not left or not right:
#     print(1)
# else:
#     print(2)
"""


# 4
"""
from math import sqrt
from math import pi
from math import exp

# 将dataset依据结果值分类，并将各样本依分类放入字典，最后返回这个字典。
def split_our_data_by_class(dataset):
    splited_data = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in splited_data):
            splited_data[class_value] = list()
        splited_data[class_value].append(vector)
    print(splited_data)
    return splited_data

def calculate_the_mean(a_list_of_num):
    mean = sum(a_list_of_num) / float(len(a_list_of_num))
    return mean

def calculate_the_standard_deviation(a_list_of_num):
    the_mean = calculate_the_mean(a_list_of_num)
    the_variance = sum([(x - the_mean) ** 2 for x in a_list_of_num]) / float(len(a_list_of_num) - 1)
    std = sqrt(the_variance)
    return std

def describe_our_data(dataset):
    description = [(calculate_the_mean(column),
                    calculate_the_standard_deviation(column),
                    len(column)) for column in zip(*dataset)]
    a = zip(*dataset)
    # print(list(a))
    # print(description)
    del (description[-1]) # 将对结果column的description删除，保留对各样本属性culumn的description
    return description # 平均值、标准差、数量

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

# splited_data = split_our_data_by_class(dataset)
# description = describe_our_data(dataset)
# print(description)

# a = zip(*dataset)
# b = zip(dataset)

# print(list(a))
# print(b)
"""


# 5.绝对值
"""
a = -1
b = abs(a)
print(b)
"""

