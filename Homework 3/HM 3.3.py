import random

def convolve(image_matrix, filter_matrix):
    """
    Функция для одноканальной свертки изображения (свертки одного канала)
    """
    m = len(image_matrix)
    n = len(image_matrix[0]) if m > 0 else 0

    k = len(filter_matrix)
    l = len(filter_matrix[0]) if k > 0 else 0

    result_m = m - k + 1
    result_n = n - l + 1

    if result_m <= 0 or result_n <= 0:
        return []

    result = [[0] * result_n for _ in range(result_m)]

    for i in range(result_m):
        for j in range(result_n):
            acc = 0
            for ki in range(k):
                for kj in range(l):
                    acc += image_matrix[i + ki][j + kj] * filter_matrix[ki][kj]
            result[i][j] = acc

    return result
def filter(func):
    """
    Декоратор для мультиканальной свертки изображения (свертки трех каналов) [M, N, C].
    Позволяет применять одноканальную функцию свертки поканально.
    """
    
    def wrapper(image, *argv):
        """
        Функция-обёртка, которая обрабатывает каждый канал отдельно.
        image - [M, N, 3])
        """
        M = len(image)
        N = len(image[0]) if M > 0 else 0
        C = len(image[0][0]) if M > 0 and N > 0 else 0
        
        if C == 0:
            return []

        channels_filtered = []
        
        for c in range(C):
            channel = [[image[i][j][c] for j in range(N)] for i in range(M)]
            
            filtered_channel = func(channel, *argv)
            
            channels_filtered.append(filtered_channel)
        
        M_out = len(channels_filtered[0]) if channels_filtered else 0
        N_out = len(channels_filtered[0][0]) if M_out > 0 else 0
        
        result = [[[channels_filtered[c][i][j] 
                    for c in range(C)] 
                   for j in range(N_out)] 
                  for i in range(M_out)]
        
        return result
    
    return wrapper
#пример
M, N, C = 7, 7, 3
image = [[[random.uniform(0, 1) for _ in range(C)] 
          for _ in range(N)] 
         for _ in range(M)]

print("Многоканальное изображение:")
print(f"Размер: {len(image)}x{len(image[0])}x{len(image[0][0])}")

filter = [[0.7, 0.7, 0.7],
          [0.7, 0.7, 0.7],
          [0.7, 0.7, 0.7]]
print("Фильтр:")
for row in filter:
    for num in row:
        print(num, end = " ")
    print("")

result = (image, filter)

M_out = len(result)
N_out = len(result[0])
C_out = len(result[0][0])

print(f"Результат: {M_out}x{N_out}x{C_out}")