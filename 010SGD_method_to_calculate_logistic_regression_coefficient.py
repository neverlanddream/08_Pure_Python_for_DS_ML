# 用SGD计算逻辑回归参数
# 整个09中的各函数，并用数据试一下。
from math import exp


def prediction(row, coeffients):            #yhat先赋值给b_0,然后通过+=b_1*x_1,...(循环)...,+=b_k*x_k，最后算出最终的yhat。
    yhat = coeffients[0]                    #coeffients[0]==b_0 , coeffients[1]==b_1 ,..., coeffients[k]==b_k
    for i in range(len(row)-1):             #len(row)计算出的是序数，len(row)-1转换为基数，即机器识别列表元素相位的数
        yhat += coeffients[i+1] * row[i]    #row[0]==x_1 , row[1]==x_2 ,..., row[k-1]==x_k, row[k]==x_k+1
    return 1/(1+exp(-yhat))


def using_sgd_method_to_calculate_coefficients(training_dataset, learning_rate, n_times_epoch):
    coefficients = [0.0 for i in range(len(training_dataset[0]))]
    for epoch in range(n_times_epoch):
        the_sum_of_error = 0
        for row in training_dataset:
            y_hat = prediction(row, coefficients)
            error = y_hat - row[-1]
            the_sum_of_error += error ** 2
            gradient = y_hat * ( 1.0 - y_hat )
            # 先计算b_0,再用for计算其余系数b
            coefficients[0] = coefficients[0] - learning_rate * error * gradient
            for i in range(len(row)-1):
                coefficients[i+1] = coefficients[i+1] - learning_rate * error * gradient * row[i]
        print("This is epoch 【%d】 , the learning rate we are using is 【%.3f】, the error is  【%.3f】"%(
            epoch,learning_rate,the_sum_of_error))
    return coefficients



dataset = [[2 , 2 , 0],
           [2 , 4 , 0],
           [3 , 3 , 0],
           [4 , 5 , 0],
           [8 , 1 , 1],
           [8.5,3.5,1],
           [9 , 1 , 1],
           [10, 4 , 1]]

learning_rate = 0.1
n_times_epoch = 1000

coef = using_sgd_method_to_calculate_coefficients(dataset, learning_rate, n_times_epoch,)
b_0, b_1, b_2 = coef
# print(coef)
print("b_0 = %.3f, b_1 = %.3f, b_2 = %.3f " %(b_0, b_1, b_2 ))
# b_0 = -0.593, b_1 = 0.923, b_2 = -1.421   100次的结果
# b_0 = -1.156, b_1 = 1.483, b_2 = -2.308   1000次的结果