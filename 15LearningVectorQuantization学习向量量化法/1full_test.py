

from math import sqrt
from random import randrange


# 1
def calculate_euclidean_distance(row1,row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# 2 codebooks里的每一个向量（与样本结构相同），分别与test_row计算距离，并把距离最近的codebook vector返回，作为BMU
def calculate_BMU(codebooks, test_row): # 数据类型：codebooks列表的列表，test_row列表
    distances = list()
    for codebook in codebooks: # 数据类型：codebook列表
        dist = calculate_euclidean_distance(codebook,test_row)  # 数据类型：dist数值
        distances.append((codebook,dist))  # 数据类型：distances元组的列表,[([codebook],dist),([codebook],dist),...]
    distances.sort(key=lambda every_tuple : every_tuple[1]) # 以dist进行排序
    return distances[0][0]  # 第一个0是顺位上第一个，第二个0是将元组中的codebook部分拿出来。返回值为[codebook vector]

# 3
def make_random_codebook(train):
    n_index = len(train)
    # print(n_index)
    n_features = len(train[0])
    # print(n_features)
    #train[][]中,每一次循环得出i值，按左右顺序执行一遍[][]。本例中，i=0时，[第1次随机][1]；i=1时，[第2次随机][2]；i=2时，[第3次随机][3]。
    #本例，相当于先在第1个属性里，从10个样本随机取一个值；然后在第2个属性里，从10个样本随机取一个值；最后在第3个属性里，从10个样本随机取一个值，赋值给codebook.
    codebook = [train[randrange(n_index)][i] for i in range(n_features)]
    # print(codebook)
    return codebook

# 4
def train_codebooks(train,n_codebooks,learn_rate,epochs):
    codebooks = [make_random_codebook(train) for i in range(n_codebooks)]
    print("codebooks_0",codebooks)
    for epoch in range(epochs):
        rate = learn_rate * (1-(epoch/float(epochs)))
        sum_error = 0.0
        for row in train: #在train中遍历每个样本，更新bmu。外层还需要迭代epochs次数，一直在更新bmu中各属性。
            bmu = calculate_BMU(codebooks,row) #codebooks中与当前循环到的row中距离最近的为bmu
            # print("bmu_1",bmu)
            # print("codebooks_1", codebooks)
            for i in range(len(row)-1): #遍历bmu中的每个属性，下面对比row和bmu该属性的差异，并通过rate调整bmu的该属性
                error = row[i] - bmu[i]
                sum_error += error**2
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
            # print("bmu_2", bmu)
            # print("codebooks_2", codebooks) #？？？为什么每次更新bmu中的属性值后，codebooks也会更新？是因为calculate_BMU里面使用元组吗？？？

        print('Our current epoch is【%d】, Our current learning rate is :【%.3f】, Our current sum of errors is 【%.3f】' % (epoch,rate,sum_error))
    return codebooks


# 5
dataset = [[1.80,1.91,0],
           [1.85,2.11,0],
           [2.31,2.88,0],
           [3.54,-3.21,0],
           [3.66,3.12,0],
           [5.52,2.13,1],
           [6.32,1.46,1],
           [7.35,2.34,1],
           [7.78,3.26,1],
           [8.43,-0.34,1]
           ]
learning_rate = 0.5
n_epoch = 10
n_codebooks = 2 #当多次运行时，sum of errors 可能会一直增加，原因很可能是因为样本数据过少和n_codebooks过小。
codebooks = train_codebooks(dataset,n_codebooks,learning_rate,n_epoch)
print('Our codebook is : %s' % codebooks)