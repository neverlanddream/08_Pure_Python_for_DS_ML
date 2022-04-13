"""一些特殊用法、技巧的备注"""


# 1.关于用sum对列表求和降维的用法
sum_for_list_v1 = [[2 , 2 , 0],
                   [2 , 4 , 0],
                   [9 , 1 , 1],
                   [10, 4 , 1]]

# sum_for_list_v2 = sum( sum_for_list_v1 , [] )
# print(sum_for_list_v2)

#不和空列表求和会报错
# sum_for_list_v3 = sum( sum_for_list_v1 )
# print(sum_for_list_v3)


# 2.将string型样本属性数据和string型样本结果数据转化成浮点型。
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

"""
enumerate在字典上是枚举、列举的意思
对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
关于enumerate的用法可参考 https://www.jb51.net/article/177995.htm
"""

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


# 3.查看非空，or
left = []
right = []

if not left or not right:
    print(1)
else:
    print(2)