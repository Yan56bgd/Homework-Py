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