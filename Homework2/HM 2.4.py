# 4. Умножение матрицы на вектор
def task4(matrix, vector):
    result = []
    for i in range(len(matrix)):
        sum_val = 0
        for j in range(len(vector)):
            sum_val += matrix[i][j] * vector[j]
        result.append(sum_val)
    return result