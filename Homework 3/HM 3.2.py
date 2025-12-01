import random

# a. Создание вектора
def create_vector(N):
    return [random.random() for _ in range(N)]

# b. Создание матрицы
def create_matrix(M, N):
    return [[random.random() for _ in range(N)] for _ in range(M)]

# c. Умножение матрицы на вектор
def matrix_vector_mult(matrix, vector):
    result = []
    for i in range(len(matrix)):
        sum_val = 0
        for j in range(len(vector)):
            sum_val += matrix[i][j] * vector[j]
        result.append(sum_val)
    return result

# d. Печать матрицы
def print_matrix(matrix):
    for row in matrix:
        print([f"{x:.2f}" for x in row])

# e. Печать вектора
def print_vector(vector):
    print([f"{x:.2f}" for x in vector])

# f. Сумма диагональных элементов
def diagonal_sum(matrix):
    total = 0
    for i in range(min(len(matrix), len(matrix[0]))):
        total += matrix[i][i]
    return total

# g. Двумерная свертка
def convolution_2d(image, kernel):
    M, N = len(image), len(image[0])
    K = len(kernel)
    half = K // 2
    result = [[0 for _ in range(N)] for _ in range(M)]
    
    for i in range(M):
        for j in range(N):
            sum_val = 0
            for k in range(K):
                for l in range(K):
                    x = i + k - half
                    y = j + l - half
                    if 0 <= x < M and 0 <= y < N:
                        sum_val += image[x][y] * kernel[k][l]
            result[i][j] = sum_val
    return result

# Демонстрация всех функций
print("=== ДЕМОНСТРАЦИЯ ВСЕХ ФУНКЦИЙ ===\n")

print("a. Создание вектора длины 5:")
vector = create_vector(5)
print_vector(vector)

print("\nb. Создание матрицы 3x4:")
matrix = create_matrix(3, 4)
print_matrix(matrix)

print("\nc. Умножение матрицы 3x3 на вектор длины 3:")
matrix3x3 = create_matrix(3, 3)
vector3 = create_vector(3)
print("Матрица:")
print_matrix(matrix3x3)
print("Вектор:")
print_vector(vector3)
mult_result = matrix_vector_mult(matrix3x3, vector3)
print("Результат умножения:")
print_vector(mult_result)

print("\nd. Печать матрицы 2x2:")
small_matrix = create_matrix(2, 2)
print_matrix(small_matrix)

print("\ne. Печать вектора длины 4:")
vector4 = create_vector(4)
print_vector(vector4)

print("\nf. Сумма диагональных элементов матрицы 3x3:")
square_matrix = create_matrix(3, 3)
print("Матрица:")
print_matrix(square_matrix)
diag_sum = diagonal_sum(square_matrix)
print(f"Сумма диагональных элементов: {diag_sum:.2f}")

print("\ng. Двумерная свертка изображения 4x4 с ядром 3x3:")
image = create_matrix(4, 4)
kernel = [[0.1, 0.2, 0.1], 
          [0.2, 0.4, 0.2], 
          [0.1, 0.2, 0.1]]
print("Изображение:")
print_matrix(image)
print("Ядро свертки:")
print_matrix(kernel)
conv_result = convolution_2d(image, kernel)
print("Результат свертки:")
print_matrix(conv_result)





            #Листинг:
#a. Создание вектора длины 5:
#['0.88', '0.42', '0.59', '0.12', '0.88']

#b. Создание матрицы 3x4:
#['0.71', '0.85', '0.22', '0.78']
#['0.78', '0.68', '0.56', '0.81']
#['0.27', '0.32', '0.64', '0.65']

#c. Умножение матрицы 3x3 на вектор длины 3:
#Матрица:
#['0.26', '0.64', '0.62']
#['0.58', '0.04', '1.00']
#['0.66', '0.90', '0.06']
#Вектор:
#['0.74', '0.21', '0.29']
#Результат умножения:
#['0.51', '0.72', '0.69']

#d. Печать матрицы 2x2:
#['0.52', '0.39']
#['0.78', '0.47']

#e. Печать вектора длины 4:
#['0.72', '0.58', '0.17', '0.18']

#f. Сумма диагональных элементов матрицы 3x3:
#Матрица:
#['0.34', '0.33', '0.14']
#['0.11', '0.86', '0.12']
#['0.76', '0.07', '0.05']
#Сумма диагональных элементов: 1.25

#g. Двумерная свертка изображения 4x4 с ядром 3x3:
#Изображение:
#['0.41', '0.47', '0.20', '0.21']
#['0.74', '0.25', '0.02', '0.33']
#['0.63', '0.27', '0.54', '0.47']
#['0.14', '0.79', '0.33', '0.78']
#Ядро свертки:
#['0.10', '0.20', '0.10']
#['0.20', '0.40', '0.20']
#['0.10', '0.20', '0.10']
#Результат свертки:
#['0.43', '0.44', '0.28', '0.19']
#['0.63', '0.58', '0.42', '0.35']
#['0.59', '0.68', '0.65', '0.56']
#['0.37', '0.58', '0.63', '0.53']