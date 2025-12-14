# matrix_operations.py
import time
import random
import math
from typing import List, Tuple, Optional
import sys


# 1. Умножение «матрица - матрица»
def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Умножение матрицы A на матрицу B"""
    if not A or not B:
        return []
    
    rows_a = len(A)
    cols_a = len(A[0])
    rows_b = len(B)
    cols_b = len(B[0])
    
    if cols_a != rows_b:
        raise ValueError("Невозможно умножить матрицы: несовпадающие размеры")
    
    # Создаем результирующую матрицу
    result = [[0.0] * cols_b for _ in range(rows_a)]
    
    # Умножение матриц
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += A[i][k] * B[k][j]
    
    return result


# 2. Умножение «матрица - вектор»
def matrix_vector_multiply(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Умножение матрицы на вектор"""
    if not matrix or not vector:
        return []
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    if cols != len(vector):
        raise ValueError("Невозможно умножить матрицу на вектор: несовпадающие размеры")
    
    result = [0.0] * rows
    
    for i in range(rows):
        for j in range(cols):
            result[i] += matrix[i][j] * vector[j]
    
    return result


# 3. Расчет следа матрицы
def matrix_trace(matrix: List[List[float]]) -> float:
    """Вычисление следа квадратной матрицы"""
    if not matrix:
        return 0.0
    
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("Матрица должна быть квадратной для вычисления следа")
    
    trace = 0.0
    for i in range(n):
        trace += matrix[i][i]
    
    return trace


# 4. Скалярное произведение двух векторов
def dot_product(v1: List[float], v2: List[float]) -> float:
    """Скалярное произведение векторов"""
    if len(v1) != len(v2):
        raise ValueError("Векторы должны иметь одинаковую длину")
    
    result = 0.0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    
    return result


# 5. Расчет гистограммы для вектора с изменяемым количеством квантов
def calculate_histogram(data: List[float], num_bins: int = 10) -> Tuple[List[int], List[float]]:
    """Вычисление гистограммы с заданным количеством бинов"""
    if not data:
        return [], []
    
    min_val = min(data)
    max_val = max(data)
    
    if min_val == max_val:
        bin_edges = [min_val - 0.5, min_val + 0.5]
        hist = [len(data)]
        return hist, bin_edges
    
    bin_width = (max_val - min_val) / num_bins
    bin_edges = [min_val + i * bin_width for i in range(num_bins + 1)]
    
    histogram = [0] * num_bins
    
    for value in data:
        if value == max_val:
            histogram[num_bins - 1] += 1
        else:
            bin_index = int((value - min_val) / bin_width)
            histogram[bin_index] += 1
    
    return histogram, bin_edges


# 6. Фильтрация вектора ядерным фильтром
def kernel_filter(data: List[float], kernel: List[float]) -> List[float]:
    """Фильтрация вектора с помощью ядра (например, [-1, 0, 1] для градиента)"""
    if not data:
        return []
    
    n = len(data)
    k = len(kernel)
    k_half = k // 2
    
    extended = [0.0] * (n + k - 1)
    
    for i in range(n):
        extended[i + k_half] = data[i]
    
    for i in range(k_half):
        extended[i] = data[k_half - i - 1]
    
    for i in range(k_half):
        extended[n + k_half + i] = data[n - i - 1]
    
    result = [0.0] * n
    for i in range(n):
        for j in range(k):
            result[i] += extended[i + j] * kernel[j]
    
    return result


# 7. Чтение/запись данных в файл
def write_matrix_to_file(matrix: List[List[float]], filename: str):
    """Запись матрицы в файл"""
    with open(filename, 'w') as f:
        f.write(f"{len(matrix)} {len(matrix[0])}\n")
        for row in matrix:
            f.write(" ".join(str(x) for x in row) + "\n")


def read_matrix_from_file(filename: str) -> List[List[float]]:
    """Чтение матрицы из файла"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return []
    
    sizes = list(map(int, lines[0].strip().split()))
    if len(sizes) != 2:
        raise ValueError("Неверный формат файла: ожидалось два числа в первой строке")
    
    rows, cols = sizes
    matrix = []
    
    for i in range(1, min(rows + 1, len(lines))):
        row_data = list(map(float, lines[i].strip().split()))
        if len(row_data) != cols:
            row_data += [0.0] * (cols - len(row_data))
        matrix.append(row_data[:cols])  # Берем только cols элементов
    
    while len(matrix) < rows:
        matrix.append([0.0] * cols)
    
    return matrix


def write_vector_to_file(vector: List[float], filename: str):
    """Запись вектора в файл"""
    with open(filename, 'w') as f:
        f.write(f"{len(vector)}\n")
        f.write(" ".join(str(x) for x in vector))


def read_vector_from_file(filename: str) -> List[float]:
    """Чтение вектора из файла"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return []
    
    try:
        size = int(lines[0].strip())
        if len(lines) > 1:
            vector = list(map(float, lines[1].strip().split()))
            return vector[:size]
        else:
            return []
    except ValueError:
        vector = list(map(float, lines[0].strip().split()))
        return vector


def generate_random_matrix(rows: int, cols: int) -> List[List[float]]:
    """Генерация случайной матрицы"""
    return [[random.uniform(-10, 10) for _ in range(cols)] for _ in range(rows)]


def generate_random_vector(size: int) -> List[float]:
    """Генерация случайного вектора"""
    return [random.uniform(-10, 10) for _ in range(size)]


def measure_time(func, *args, **kwargs):
    """Измерение времени выполнения функции"""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


def run_benchmark():
    """Запуск бенчмарков и сохранение результатов в файл"""
    results = []
    
    test_cases = [
        (5, 5),   # маленькие
        (10, 10), # средние
        (20, 20), # большие
        (50, 50), # очень большие
    ]
    
    for rows, cols in test_cases:
        print(f"\nТест для матрицы {rows}x{cols}:")
        
        A = generate_random_matrix(rows, cols)
        B = generate_random_matrix(cols, rows)
        v = generate_random_vector(cols)
        
        # 1. Умножение матрица-матрица
        _, time_matmul = measure_time(matrix_multiply, A, B)
        results.append(f"Матрица {rows}x{cols} * {cols}x{rows}: {time_matmul:.6f} сек")
        print(f"  Умножение матриц: {time_matmul:.6f} сек")
        
        # 2. Умножение матрица-вектор
        _, time_matvec = measure_time(matrix_vector_multiply, A, v)
        results.append(f"Матрица {rows}x{cols} * вектор {cols}: {time_matvec:.6f} сек")
        print(f"  Умножение матрица-вектор: {time_matvec:.6f} сек")
        
        # 3. След матрицы (только для квадратных)
        if rows == cols:
            _, time_trace = measure_time(matrix_trace, A)
            results.append(f"След матрицы {rows}x{rows}: {time_trace:.6f} сек")
            print(f"  След матрицы: {time_trace:.6f} сек")
        
        # 4. Скалярное произведение
        v2 = generate_random_vector(cols)
        _, time_dot = measure_time(dot_product, v, v2)
        results.append(f"Скалярное произведение векторов {cols}: {time_dot:.6f} сек")
        print(f"  Скалярное произведение: {time_dot:.6f} сек")
        
        # 5. Гистограмма
        _, time_hist = measure_time(calculate_histogram, v, 10)
        results.append(f"Гистограмма вектора {cols}: {time_hist:.6f} сек")
        print(f"  Гистограмма: {time_hist:.6f} сек")
        
        # 6. Фильтрация
        kernel = [-1, 0, 1]
        _, time_filter = measure_time(kernel_filter, v, kernel)
        results.append(f"Фильтрация вектора {cols}: {time_filter:.6f} сек")
        print(f"  Фильтрация: {time_filter:.6f} сек")
    
    with open("benchmark_results.txt", "w") as f:
        f.write("Результаты измерений времени выполнения операций:\n")
        f.write("=" * 60 + "\n")
        for result in results:
            f.write(result + "\n")
    
    print("\nРезультаты сохранены в файл 'benchmark_results.txt'")


if __name__ == "__main__":
    print("Демонстрация работы функций из matrix_operations.py")
    print("=" * 60)
    
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    v = [1, 2, 3]
    
    print("Матрица A (2x3):", A)
    print("Матрица B (3x2):", B)
    print("Вектор v:", v)
    
    # 1. Умножение матрица-матрица
    C = matrix_multiply(A, B)
    print("\n1. Умножение матрица-матрица (A * B):", C)
    
    # 2. Умножение матрица-вектор
    result_vec = matrix_vector_multiply(A, v)
    print("2. Умножение матрица-вектор (A * v):", result_vec)
    
    # 3. След матрицы (квадратной)
    square_matrix = [[1, 2], [3, 4]]
    trace = matrix_trace(square_matrix)
    print("3. След матрицы [[1,2],[3,4]]:", trace)
    
    # 4. Скалярное произведение
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    dot = dot_product(v1, v2)
    print(f"4. Скалярное произведение {v1} и {v2}:", dot)
    
    # 5. Гистограмма
    data = [1.2, 2.3, 3.4, 1.5, 2.6, 3.7, 1.8, 2.9, 3.0]
    hist, bins = calculate_histogram(data, 3)
    print(f"5. Гистограмма для {data}:")
    print(f"   Биньі: {bins}")
    print(f"   Значения: {hist}")
    
    # 6. Фильтрация
    data_vec = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    kernel = [-1, 0, 1]
    filtered = kernel_filter(data_vec, kernel)
    print(f"6. Фильтрация {data_vec} ядром {kernel}:", filtered)
    
    # 7. Тест чтения/записи
    test_matrix = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
    write_matrix_to_file(test_matrix, "test_matrix.txt")
    loaded_matrix = read_matrix_from_file("test_matrix.txt")
    print(f"\n7. Тест записи/чтения матрицы:")
    print(f"   Исходная: {test_matrix}")
    print(f"   Загруженная: {loaded_matrix}")
    
    # Запуск бенчмарков
    print("\n" + "=" * 60)
    print("Запуск бенчмарков...")
    run_benchmark()