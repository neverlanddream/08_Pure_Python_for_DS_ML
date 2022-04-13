"""依据gini系数进行一次分类v2
思路：
将整个dataset依据最后一列结果分成classes，一般是两类结果。不考虑最后的结果列，依据列（维度、属性）数量，开始遍历各列（维度、属性）。
    在当前列（以index形式），遍历各行。
        在当前行（某个单一样本数据），依据当前列，确定该列的一个值（某个属性的具体值），然后依据这个值将dataset分成classes个（2个）组（groups）。
        具体为执行test_split（），遍历dataset各行，当遍历的样本对应列（属性）小于当前行的列值（属性）时，放入一组，否则放入另一个组.
            依据classes结果（如只有2类），计算各组gini系数，最终计算该分类gini系数。
    得到所有属性对应不同属性值时的gini。
选全局最小gini对应情况的分类：哪个属性，该属性对应何数值区分。完成一次分类，确定root node。
与2.1代码一致，代码注释更简介。
"""


def dataset_test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def calculate_the_gini_index(groups, classes):
    # 计算有多少实例
    n_instances = float(sum([len(group) for group in groups]))
    # 把每一个group里面的加权gini计算出来
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # *注意，这里不能除以0，所以我们要考虑到分母为0的情况
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # 这个做了一个加权处理
        gini += (1 - score) * (size / n_instances)

    return gini


def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    posi_index, posi_value, posi_score, posi_groups = 888, 888, 888, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = dataset_test_split(index, row[index], dataset)
            gini = calculate_the_gini_index(groups, class_values)
            print("X%d < %.3f Gini=%.3f" % ((index + 1), row[index], gini))
            if gini < posi_score:
                posi_index, posi_value, posi_score, posi_groups = index, row[index], gini, groups
    return {'index': posi_index, 'value': posi_value, 'groups': posi_groups}


# 示例1：
dataset = [[2.1, 1.1, 0],
           [3.4, 2.5, 0],
           [1.3, 5.8, 0],
           [1.9, 8.6, 0],
           [3.7, 6.2, 0],
           [8.8, 1.1, 1],
           [9.6, 3.4, 1],
           [10.2, 7.4, 1],
           [7.7, 8.8, 1],
           [9.7, 6.9, 1]]

split = get_split(dataset)
print('Split:[X%d < %.3f]' % ((split['index'] + 1), split['value']))


# 示例2：
# case_for_two_classes = (['T','N','B','1'],
#                         ['T','Y','B','1'],
#                         ['T','N','A','1'],
#                         ['F','Y','B','0'],
#                         ['F','N','A','0'])
# split = get_split(case_for_two_classes)
# print(split)