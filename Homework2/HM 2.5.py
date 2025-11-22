# 5. Замена отрицательных значений
def task5(data):
    result = data.copy()
    n = len(data)
    
    for i in range(n):
        if data[i] < 0:
            left_pos = None
            right_pos = None
            
            # Поиск левого положительного
            for j in range(i-1, -1, -1):
                if data[j] > 0:
                    left_pos = data[j]
                    break
            
            # Поиск правого положительного
            for j in range(i+1, n):
                if data[j] > 0:
                    right_pos = data[j]
                    break
            
            if left_pos is not None and right_pos is not None:
                result[i] = (left_pos + right_pos) / 2
    
    return result