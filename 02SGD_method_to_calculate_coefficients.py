
def make_prediction(input_row: list, coefficients: float) -> object:
    output_y_hat: float = coefficients[0]
    for i in range(len(input_row) - 1):
        output_y_hat += coefficients[i + 1] * input_row[i]
    return output_y_hat


def using_sgd_method_to_calculate_coefficients(training_dataset, learning_rate, n_times_epoch):
    coefficients = [0.0 for i in range(len(training_dataset[0]))]
    for epoch in range(n_times_epoch):
        the_sum_of_error = 0
        for row in training_dataset:
            y_hat = make_prediction(row, coefficients)
            error = y_hat - row[-1]
            the_sum_of_error += error ** 2
            coefficients[0] = coefficients[0] - learning_rate * error
            for i in range(len(row)-1):
                coefficients[i+1] = coefficients[i+1] - learning_rate * error * row[i]
        print("This is epoch 【%d】 , the learning rate we are using is 【%.3f】, the error is  【%.3f】"%(
            epoch,learning_rate,the_sum_of_error))
    return coefficients


your_training_dataset = [[1, 1],[2, 3],[4, 3],[3, 2],[5, 5]]
your_model_learning_rate = 0.001
your_n_times_epoch = 500
your_coefficients = using_sgd_method_to_calculate_coefficients(your_training_dataset,your_model_learning_rate,
                                                               your_n_times_epoch)
print(your_coefficients)