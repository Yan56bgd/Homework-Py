def convolution_filter(data, kernel):
    result = []
    kernel_size = len(kernel)
    half = kernel_size // 2
    
    for i in range(len(data)):
        sum_val = 0
        for j in range(kernel_size):
            pos = i + j - half
            if pos < 0:
                pos = 0
            elif pos >= len(data):
                pos = len(data) - 1
            sum_val += data[pos] * kernel[j]
        result.append(sum_val)
    
    return result

data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
kernel = [0.5, 1, 0.5]

result = convolution_filter(data, kernel)

print(f"Данные: {data}")
print(f"Ядро: {kernel}")
print(f"Результат свертки: {result}")