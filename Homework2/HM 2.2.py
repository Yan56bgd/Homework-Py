# 2. Гистограмма и вероятности
def task2():
    import random
    data = [random.randint(0, 100) for _ in range(20)]
    
    histogram = [0] * 10
    for num in data:
        bin_index = num // 10
        histogram[bin_index] += 1
    
    probabilities = [count / len(data) for count in histogram]
    
    return data, histogram, probabilities