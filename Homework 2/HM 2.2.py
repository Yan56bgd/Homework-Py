import random

data = [random.randint(0, 100) for _ in range(20)]

histogram = [0] * 10
probabilities = [0] * 10

for value in data:
    bin_index = value // 10
    if bin_index > 9:
        bin_index = 9
    histogram[bin_index] += 1

for i in range(10):
    probabilities[i] = histogram[i] / len(data)

print("Список данных:", data)
print("\nГистограмма:")
for i in range(10):
    start = i * 10
    end = start + 9
    if i == 9:
        end = 100
    print(f"[{start:2}-{end:3}]: {histogram[i]:2} элементов")

print("\nВероятности:")
for i in range(10):
    start = i * 10
    end = start + 9
    if i == 9:
        end = 100
    print(f"[{start:2}-{end:3}]: {probabilities[i]:.2f}")