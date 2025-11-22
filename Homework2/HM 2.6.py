# 6. Сверточная фильтрация
def task6(data, kernel):
    kernel_size = len(kernel)
    half_kernel = kernel_size // 2
    result = []
    
    for i in range(len(data)):
        sum_val = 0
        for j in range(kernel_size):
            data_index = i + j - half_kernel
            if 0 <= data_index < len(data):
                sum_val += data[data_index] * kernel[j]
        result.append(sum_val)
    
    return result