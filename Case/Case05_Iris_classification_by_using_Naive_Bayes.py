"""用朴素贝叶斯对鸢尾花分类
数据集信息：
这可能是模式识别文献中最著名的数据库。Fisher 的论文是该领域的经典之作，至今仍被频繁引用。（例如，参见 Duda & Hart。）
数据集包含 3 个类别，每个类别 50 个实例，其中每个类别指的是一种鸢尾植物。一类与另一类线性可分；后者不能彼此线性分离。
预测属性：鸢尾植物类。
这是一个非常简单的域。
该数据不同于 Fishers 文章中提供的数据（由 Steve Chadwick 识别， spchadwick '@' espeedaz.net）。
第 35 个样本应该是：4.9,3.1,1.5,0.2,"Iris-setosa" 其中错误在第四个特征中。
第 38 个样本：4.9,3.6,1.4,0.1，“Iris-setosa”，其中错误在第二个和第三个特征中。

属性信息：
1. 萼片长度 cm
2. 萼片宽度 cm
3. 花瓣长度 cm
4. 花瓣宽度 cm
5. 类别：
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica

"""


from csv import reader
from random import randrange
from math import sqrt
from math import exp
from math import pi


# 1.csv reader helper function
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# dataset = load_csv('iris.csv')
# print(dataset)

# 2 处理属性值
def convert_str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# 3 处理结果值【这部分代码反复看】
def convert_str_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique_value = set(class_values)
    look_up = dict()
    for i, value in enumerate(unique_value):
        look_up[value] = i
    for row in dataset:
        row[column] = look_up[row[column]]
    return look_up

# 4 交叉验证
def n_fold_cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# 5 计算准确性
def calculate_our_model_accuracy(actual, predicted):
    correct_count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct_count += 1
    return correct_count / float(len(actual)) * 100.0

# 6 评价模型准确性得分
def whether_our_model_is_good_or_not(dataset, algo, n_folds, *args):
    folds = n_fold_cross_validation_split(dataset, n_folds)
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
        accuracy = calculate_our_model_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores

# 7 以字典形式储存数据。以下开始建立模型。
def split_our_data_by_class(dataset):
    splited = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in splited):
            splited[class_value] = list()
        splited[class_value].append(vector)
    return splited

# 8
def mean(a_list_of_numbers):
    return sum(a_list_of_numbers) / float(len(a_list_of_numbers))

# 9
def stdev(a_list_of_numbers):
    the_mean_of_a_list_numbers = mean(a_list_of_numbers)
    variance = sum([(x - the_mean_of_a_list_numbers) ** 2 for x in a_list_of_numbers]) / float(
        len(a_list_of_numbers) - 1)
    return sqrt(variance)

# 10 编写类似numpy中describe的代码，用于事先对数据的了解。
def describe_our_data(dataset):
    description = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (description[-1])
    return description

# 11
def describe_our_data_by_class(dataset):
    data_split = split_our_data_by_class(dataset)
    description = dict()
    for class_value, rows in data_split.items():
        description[class_value] = describe_our_data(rows)
    return description # 返回字典{分类：[(mean, std, count),(),...] ,分类：[], ... }，其中每一个（）元组是对应一个属性

# 12 假设x属于正态分布，以此计算x的概率
def calculate_the_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    result = (1 / (sqrt(2 * pi) * stdev)) * exponent
    return result

# 13 返回P(样本属性|分类)，并按照分类情况以字典存在probabilities
def calculate_the_probability_by_class(description, row): # row对应测试集中的每一个样本的X（属性）部分（不含结果值部分）
    total_rows = sum([description[label][0][2] for label in description])
    probabilities = dict()
    for class_value, class_description, in description.items():
        probabilities[class_value] = description[class_value][0][2] / float(total_rows)
        for i in range(len(class_description)): # 遍历字典中每个分类下的各元组，即对应每个属性的(mean, std, count)
            mean, stdev, count = class_description[i]
            probabilities[class_value] *= calculate_the_probability(row[i], mean, stdev) # 此处重点。
    return probabilities # P(该分类的概率；初始值)*P(在正态分布下，row样本每个属性i的概率；连乘)=P(该分类下，row样本对应当前属性的概率)
                         # 可以理解为讲课时“你好吗钱”示例：返回值为P(‘你好吗钱’|N)和P(‘你好吗钱’|S)，并以字典形式存在probabilities中。

# 14 算法核心是“选择”概率最大情况对应的分类
def predict(description, row):
    probabilities = calculate_the_probability_by_class(description, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob: # 这里是算法“选择”的核心，依据13计算各标签概率的最大值，选样本row的label
            best_prob = probability
            best_label = class_value
    return best_label

# 15 算法存在意义是返回预测值
def naive_bayes(train, test):
    description = describe_our_data_by_class(train)
    predictions = list()
    for row in test: # row是测试集中的每一个样本
        prediction = predict(description, row)
        predictions.append(prediction)
    return predictions # 返回算法的一系列预测值；数据类型为列表[ , , ...]。

# 16
dataset = load_csv('C:/Users/MI/William_DataSci/08_Pure_Python_for_DS_ML/Base_Data/iris.csv')
print(dataset)
print('--------------------')

for i in range(len(dataset[0]) - 1):
    convert_str_to_float(dataset, i)
print(dataset)
print('--------------------')

convert_str_to_int(dataset, len(dataset[0]) - 1)
print(dataset)
print('--------------------')

n_folds = 10
scores = whether_our_model_is_good_or_not(dataset, naive_bayes, n_folds)

print("The score of our model is : 【 %s 】" % scores)
print('The accuracy of our model is : %.6f%% ' % (sum(scores) / float(len(scores))))
