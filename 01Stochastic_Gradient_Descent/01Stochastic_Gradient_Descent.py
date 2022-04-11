# make prediction （coefficients[0]即为两个参数中的b_0, coefficients[i+1]即为b_1）
# 调试状态，按F8可以逐行调试（步过）
def make_prediction(input_row: list, coefficients: float) -> object:
    output_y_hat: float = coefficients[0]
    for i in range(len(input_row) - 1):
        output_y_hat += coefficients[i + 1] * input_row[i]
    return output_y_hat


test_dataset = [[1, 1],
                [2, 3],
                [4, 3],
                [3, 2],
                [5, 5]]
test_coefficients = [0.4, 0.8]

for row in test_dataset:
    y_hat = make_prediction(row, test_coefficients)
    print("Expected = %.3f , Our_Prediction = %.3f" % (row[-1], y_hat))
