# 3. Операции с векторами
def task3():
    import random
    import math
    
    n = int(input("Введите размерность векторов: "))
    scalar = float(input("Введите скаляр: "))
    
    vector1 = [random.random() for _ in range(n)]
    vector2 = [random.random() for _ in range(n)]
    
    sum_vector = [vector1[i] + vector2[i] for i in range(n)]
    mul_vector = [vector1[i] * vector2[i] for i in range(n)]
    
    norm1 = math.sqrt(sum(x*x for x in vector1))
    norm2 = math.sqrt(sum(x*x for x in vector2))
    
    if norm1 > norm2:
        scaled_vector = [x * scalar for x in vector1]
    else:
        scaled_vector = [x * scalar for x in vector2]
    
    return sum_vector, mul_vector, scaled_vector