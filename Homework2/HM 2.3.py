import random
import math

N = int(input("Введите размерность векторов: "))
scalar = float(input("Введите скаляр: "))

vector1 = [random.randint(1, 10) for _ in range(N)]
vector2 = [random.randint(1, 10) for _ in range(N)]

sum_vector = [vector1[i] + vector2[i] for i in range(N)]
mult_vector = [vector1[i] * vector2[i] for i in range(N)]

norm1 = math.sqrt(sum(x**2 for x in vector1))
norm2 = math.sqrt(sum(x**2 for x in vector2))

if norm1 > norm2:
    scaled_vector = [x * scalar for x in vector1]
    bigger_norm_vector = vector1
else:
    scaled_vector = [x * scalar for x in vector2]
    bigger_norm_vector = vector2

print(f"Вектор 1: {vector1}")
print(f"Вектор 2: {vector2}")
print(f"Сумма: {sum_vector}")
print(f"Умножение: {mult_vector}")
print(f"Норма вектора 1: {norm1:.2f}")
print(f"Норма вектора 2: {norm2:.2f}")
print(f"Вектор с большей нормой: {bigger_norm_vector}")
print(f"Умножение на скаляр: {scaled_vector}")