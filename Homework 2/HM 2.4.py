matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

vector = [2, 1, 3]

result = []
for i in range(len(matrix)):
    sum_val = 0
    for j in range(len(vector)):
        sum_val += matrix[i][j] * vector[j]
    result.append(sum_val)

print("Матрица:")
for row in matrix:
    print(row)
print(f"Вектор: {vector}")
print(f"Результат умножения: {result}")