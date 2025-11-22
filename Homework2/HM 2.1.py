# 1. Минимальное, максимальное значение и медиана
def task1(sample):
    sorted_sample = sorted(sample)
    n = len(sorted_sample)
    min_val = sorted_sample[0]
    max_val = sorted_sample[-1]
    
    if n % 2 == 1:
        median = sorted_sample[n // 2]
    else:
        median = (sorted_sample[n // 2 - 1] + sorted_sample[n // 2]) / 2
    
    return min_val, max_val, median