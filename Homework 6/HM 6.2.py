# matrix_operations_safe.py
"""
Матричные операции с обработкой исключений
"""
import time
import random
import math
from typing import List, Tuple, Optional
import os
from exception_handler import (
    handle_file_operation, handle_data_operation, 
    validate_matrix, validate_vector, safe_execute,
    DataValidationError, FileOperationError
)

# 1. Умножение «матрица - матрица»
@handle_data_operation
def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Умножение матрицы A на матрицу B с проверкой исключений"""
    # Проверка входных данных
    if not validate_matrix(A) or not validate_matrix(B):
        raise DataValidationError("Некорректные входные матрицы")
    
    rows_a = len(A)
    cols_a = len(A[0])
    rows_b = len(B)
    cols_b = len(B[0])
    
    if cols_a != rows_b:
        raise DataValidationError(f"Невозможно умножить матрицы: несовпадающие размеры "
                                f"({cols_a} != {rows_b})")
    
    # Создаем результирующую матрицу
    result = [[0.0] * cols_b for _ in range(rows_a)]
    
    # Умножение матриц
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                # Проверка на числовые значения
                if not (isinstance(A[i][k], (int, float)) and 
                       isinstance(B[k][j], (int, float))):
                    raise DataValidationError(f"Некорректные типы элементов матриц")
                result[i][j] += A[i][k] * B[k][j]
    
    return result


# 2. Умножение «матрица - вектор»
@handle_data_operation
def matrix_vector_multiply(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Умножение матрицы на вектор с проверкой исключений"""
    # Проверка входных данных
    if not validate_matrix(matrix):
        raise DataValidationError("Некорректная входная матрица")
    
    if not validate_vector(vector):
        raise DataValidationError("Некорректный входной вектор")
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    if cols != len(vector):
        raise DataValidationError(f"Невозможно умножить матрицу на вектор: "
                                f"несовпадающие размеры ({cols} != {len(vector)})")
    
    result = [0.0] * rows
    
    for i in range(rows):
        for j in range(cols):
            # Проверка на числовые значения
            if not (isinstance(matrix[i][j], (int, float)) and 
                   isinstance(vector[j], (int, float))):
                raise DataValidationError(f"Некорректные типы элементов")
            result[i] += matrix[i][j] * vector[j]
    
    return result


# 3. Расчет следа матрицы
@handle_data_operation
def matrix_trace(matrix: List[List[float]]) -> float:
    """Вычисление следа квадратной матрицы с проверкой исключений"""
    # Проверка входных данных
    if not validate_matrix(matrix):
        raise DataValidationError("Некорректная входная матрица")
    
    n = len(matrix)
    if n != len(matrix[0]):
        raise DataValidationError("Матрица должна быть квадратной для вычисления следа")
    
    trace = 0.0
    for i in range(n):
        # Проверка на числовое значение
        if not isinstance(matrix[i][i], (int, float)):
            raise DataValidationError(f"Некорректный тип элемента на диагонали")
        trace += matrix[i][i]
    
    return trace


# 4. Скалярное произведение двух векторов
@handle_data_operation
def dot_product(v1: List[float], v2: List[float]) -> float:
    """Скалярное произведение векторов с проверкой исключений"""
    # Проверка входных данных
    if not validate_vector(v1) or not validate_vector(v2):
        raise DataValidationError("Некорректные входные векторы")
    
    if len(v1) != len(v2):
        raise DataValidationError(f"Векторы должны иметь одинаковую длину "
                                f"({len(v1)} != {len(v2)})")
    
    result = 0.0
    for i in range(len(v1)):
        # Проверка на числовые значения
        if not (isinstance(v1[i], (int, float)) and 
               isinstance(v2[i], (int, float))):
            raise DataValidationError(f"Некорректные типы элементов векторов")
        result += v1[i] * v2[i]
    
    return result


# 5. Расчет гистограммы для вектора с изменяемым количеством квантов
@handle_data_operation
def calculate_histogram(data: List[float], num_bins: int = 10) -> Tuple[List[int], List[float]]:
    """Вычисление гистограммы с заданным количеством бинов"""
    # Проверка входных данных
    if not validate_vector(data):
        raise DataValidationError("Некорректные входные данные")
    
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise DataValidationError(f"Количество бинов должно быть положительным целым числом")
    
    if num_bins > len(data):
        raise DataValidationError(f"Количество бинов не может превышать количество данных")
    
    min_val = min(data)
    max_val = max(data)
    
    if min_val == max_val:
        # Все значения одинаковы
        bin_edges = [min_val - 0.5, min_val + 0.5]
        hist = [len(data)]
        return hist, bin_edges
    
    # Вычисляем границы бинов
    bin_width = (max_val - min_val) / num_bins
    bin_edges = [min_val + i * bin_width for i in range(num_bins + 1)]
    
    # Инициализируем гистограмму
    histogram = [0] * num_bins
    
    # Считаем значения в каждом бине
    for value in data:
        if not isinstance(value, (int, float)):
            raise DataValidationError(f"Некорректный тип элемента данных")
        
        if value == max_val:
            # Максимальное значение попадает в последний бин
            histogram[num_bins - 1] += 1
        else:
            bin_index = int((value - min_val) / bin_width)
            if 0 <= bin_index < num_bins:
                histogram[bin_index] += 1
    
    return histogram, bin_edges


# 6. Фильтрация вектора ядерным фильтром
@handle_data_operation
def kernel_filter(data: List[float], kernel: List[float]) -> List[float]:
    """Фильтрация вектора с помощью ядра с проверкой исключений"""
    # Проверка входных данных
    if not validate_vector(data):
        raise DataValidationError("Некорректные входные данные")
    
    if not validate_vector(kernel):
        raise DataValidationError("Некорректное ядро фильтра")
    
    n = len(data)
    k = len(kernel)
    if k == 0:
        raise DataValidationError("Ядро фильтра не может быть пустым")
    
    k_half = k // 2
    
    # Создаем расширенный массив (зеркальное отражение границ)
    extended = [0.0] * (n + k - 1)
    
    # Копируем данные с зеркальным отражением границ
    for i in range(n):
        extended[i + k_half] = data[i]
    
    # Зеркальное отражение левой границы
    for i in range(k_half):
        extended[i] = data[k_half - i - 1]
    
    # Зеркальное отражение правой границы
    for i in range(k_half):
        extended[n + k_half + i] = data[n - i - 1]
    
    # Применяем фильтр
    result = [0.0] * n
    for i in range(n):
        for j in range(k):
            result[i] += extended[i + j] * kernel[j]
    
    return result


# 7. Чтение/запись данных в файл
@handle_file_operation
def write_matrix_to_file(matrix: List[List[float]], filename: str):
    """Запись матрицы в файл с проверкой исключений"""
    # Проверка входных данных
    if not validate_matrix(matrix):
        raise DataValidationError("Некорректная матрица для записи")
    
    # Проверка имени файла
    if not filename or not isinstance(filename, str):
        raise FileOperationError("Некорректное имя файла")
    
    # Проверка расширения файла
    if not filename.endswith('.txt'):
        print(f"Предупреждение: файл {filename} имеет нестандартное расширение")
    
    # Проверка существования файла
    if os.path.exists(filename):
        print(f"Предупреждение: файл {filename} будет перезаписан")
    
    # Запись в файл
    with open(filename, 'w', encoding='utf-8') as f:
        # Записываем размеры матрицы
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        f.write(f"{rows} {cols}\n")
        
        # Записываем данные матрицы
        for row in matrix:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")


@handle_file_operation
def read_matrix_from_file(filename: str) -> List[List[float]]:
    """Чтение матрицы из файла с проверкой исключений"""
    # Проверка существования файла
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл не найден: {filename}")
    
    # Проверка размера файла
    file_size = os.path.getsize(filename)
    if file_size == 0:
        raise FileOperationError(f"Файл пуст: {filename}")
    
    # Проверка размера файла (предотвращение чтения огромных файлов)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    if file_size > MAX_FILE_SIZE:
        raise FileOperationError(f"Файл слишком большой: {file_size} байт")
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        raise FileOperationError("Файл не содержит данных")
    
    # Первая строка содержит размеры
    try:
        sizes = list(map(int, lines[0].strip().split()))
        if len(sizes) != 2:
            raise ValueError("Неверный формат: ожидалось два числа в первой строке")
        
        rows, cols = sizes
        
        # Проверка разумности размеров
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Некорректные размеры матрицы: {rows}x{cols}")
        
        # Проверка количества строк в файле
        if len(lines) < rows + 1:
            raise ValueError(f"Недостаточно строк в файле: ожидалось {rows+1}, "
                           f"получено {len(lines)}")
        
        matrix = []
        
        # Читаем строки матрицы
        for i in range(1, rows + 1):
            if i >= len(lines):
                # Если строк меньше, чем ожидалось, создаем нулевую строку
                row_data = [0.0] * cols
            else:
                try:
                    row_data = list(map(float, lines[i].strip().split()))
                except ValueError as e:
                    raise ValueError(f"Ошибка преобразования данных в строке {i}: {e}")
            
            # Проверка количества элементов в строке
            if len(row_data) < cols:
                # Дополняем нулями
                row_data += [0.0] * (cols - len(row_data))
            elif len(row_data) > cols:
                # Обрезаем лишние элементы
                row_data = row_data[:cols]
            
            matrix.append(row_data)
        
        return matrix
        
    except ValueError as e:
        raise FileOperationError(f"Ошибка формата файла: {str(e)}")


@handle_file_operation
def write_vector_to_file(vector: List[float], filename: str):
    """Запись вектора в файл с проверкой исключений"""
    # Проверка входных данных
    if not validate_vector(vector):
        raise DataValidationError("Некорректный вектор для записи")
    
    # Проверка имени файла
    if not filename or not isinstance(filename, str):
        raise FileOperationError("Некорректное имя файла")
    
    # Проверка существования файла
    if os.path.exists(filename):
        print(f"Предупреждение: файл {filename} будет перезаписан")
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Записываем размер вектора
        f.write(f"{len(vector)}\n")
        
        # Записываем данные вектора
        f.write(" ".join(f"{x:.6f}" for x in vector))


@handle_file_operation
def read_vector_from_file(filename: str) -> List[float]:
    """Чтение вектора из файла с проверкой исключений"""
    # Проверка существования файла
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл не найден: {filename}")
    
    # Проверка размера файла
    file_size = os.path.getsize(filename)
    if file_size == 0:
        raise FileOperationError(f"Файл пуст: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        raise FileOperationError("Файл не содержит данных")
    
    try:
        # Первая строка может содержать размер (опционально)
        size_line = lines[0].strip()
        try:
            size = int(size_line)
            # Если указан размер, читаем следующую строку
            if len(lines) > 1:
                vector_data = lines[1].strip()
                vector = list(map(float, vector_data.split()))
                
                # Проверка размера
                if len(vector) != size:
                    print(f"Предупреждение: указанный размер {size} не соответствует "
                          f"фактическому {len(vector)}")
                
                return vector[:size]  # Ограничиваем указанным размером
            else:
                return []
        except ValueError:
            # Если первая строка не число, значит это данные вектора
            vector = list(map(float, size_line.split()))
            return vector
    
    except ValueError as e:
        raise FileOperationError(f"Ошибка формата файла: {str(e)}")


# Вспомогательные функции
def generate_random_matrix(rows: int, cols: int) -> List[List[float]]:
    """Генерация случайной матрицы с проверкой параметров"""
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise DataValidationError("Размеры матрицы должны быть целыми числами")
    
    if rows <= 0 or cols <= 0:
        raise DataValidationError("Размеры матрицы должны быть положительными")
    
    if rows > 1000 or cols > 1000:
        raise DataValidationError("Размеры матрицы слишком большие")
    
    return [[random.uniform(-10, 10) for _ in range(cols)] for _ in range(rows)]


def generate_random_vector(size: int) -> List[float]:
    """Генерация случайного вектора с проверкой параметров"""
    if not isinstance(size, int):
        raise DataValidationError("Размер вектора должен быть целым числом")
    
    if size <= 0:
        raise DataValidationError("Размер вектора должен быть положительным")
    
    if size > 10000:
        raise DataValidationError("Размер вектора слишком большой")
    
    return [random.uniform(-10, 10) for _ in range(size)]


def run_benchmark():
    """Запуск бенчмарков с обработкой исключений"""
    from exception_handler import safe_execute, save_error_report
    
    results = []
    test_cases = [
        (5, 5),
        (10, 10),
        (20, 20),
        (50, 50),
    ]
    
    for rows, cols in test_cases:
        print(f"\nТест для матрицы {rows}x{cols}:")
        
        try:
            # Генерируем тестовые данные
            A = generate_random_matrix(rows, cols)
            B = generate_random_matrix(cols, rows)
            v = generate_random_vector(cols)
            
            # Измеряем время выполнения для каждой операции
            operations = [
                ("Умножение матриц", matrix_multiply, (A, B)),
                ("Умножение матрица-вектор", matrix_vector_multiply, (A, v)),
                ("След матрицы", matrix_trace, (A,) if rows == cols else None),
                ("Скалярное произведение", dot_product, (v, generate_random_vector(cols))),
                ("Гистограмма", calculate_histogram, (v, 10)),
                ("Фильтрация", kernel_filter, (v, [-1, 0, 1])),
            ]
            
            for op_name, op_func, op_args in operations:
                if op_args is None:
                    continue
                
                start_time = time.perf_counter()
                try:
                    result = op_func(*op_args)
                    end_time = time.perf_counter()
                    exec_time = end_time - start_time
                    
                    results.append(f"{op_name} ({rows}x{cols}): {exec_time:.6f} сек")
                    print(f"  {op_name}: {exec_time:.6f} сек")
                    
                except Exception as e:
                    save_error_report(e, {
                        'operation': op_name,
                        'matrix_size': f"{rows}x{cols}"
                    })
                    print(f"  {op_name}: ОШИБКА - {str(e)}")
        
        except Exception as e:
            print(f"Ошибка при выполнении теста для {rows}x{cols}: {e}")
            save_error_report(e, {
                'test_case': f"{rows}x{cols}",
                'function': 'run_benchmark'
            })
            continue
    
    # Сохраняем результаты в файл
    try:
        with open("benchmark_results_safe.txt", "w", encoding='utf-8') as f:
            f.write("Результаты измерений времени выполнения операций (с обработкой исключений):\n")
            f.write("=" * 60 + "\n")
            for result in results:
                f.write(result + "\n")
        
        print("\nРезультаты сохранены в файл 'benchmark_results_safe.txt'")
    
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        save_error_report(e, {
            'function': 'run_benchmark',
            'operation': 'save_results'
        })


def test_all_operations():
    """Тестирование всех операций с обработкой исключений"""
    print("Тестирование матричных операций с обработкой исключений")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Нормальные данные',
            'matrix': [[1, 2], [3, 4]],
            'vector': [1, 2, 3],
            'should_fail': False
        },
        {
            'name': 'Пустая матрица',
            'matrix': [],
            'vector': [],
            'should_fail': True
        },
        {
            'name': 'Несовпадающие размеры',
            'matrix': [[1, 2, 3], [4, 5, 6]],
            'vector': [1, 2],  # Должно быть 3 элемента
            'should_fail': True
        },
        {
            'name': 'Некорректные типы',
            'matrix': [['a', 'b'], ['c', 'd']],
            'vector': ['x', 'y'],
            'should_fail': True
        },
    ]
    
    for test in test_cases:
        print(f"\nТест: {test['name']}")
        print(f"Матрица: {test['matrix']}")
        print(f"Вектор: {test['vector']}")
        
        try:
            if validate_matrix(test['matrix']):
                print("✓ Матрица валидна")
            else:
                print("✗ Матрица невалидна")
            
            if validate_vector(test['vector']):
                print("✓ Вектор валиден")
            else:
                print("✗ Вектор невалиден")
            
            # Пробуем выполнить операцию
            if test['matrix'] and test['vector'] and len(test['matrix'][0]) == len(test['vector']):
                result = safe_execute(
                    matrix_vector_multiply,
                    test['matrix'],
                    test['vector'],
                    error_message="Ошибка при умножении матрицы на вектор"
                )
                if result is not None:
                    print(f"Результат: {result}")
        
        except Exception as e:
            print(f"Исключение: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Демонстрация работы с исключениями
    print("Демонстрация обработки исключений в матричных операциях")
    print("=" * 60)
    
    # Тестируем операции
    test_all_operations()
    
    # Запускаем бенчмарки
    print("\n" + "=" * 60)
    print("Запуск бенчмарков...")
    run_benchmark()
    
    # Тест работы с файлами
    print("\n" + "=" * 60)
    print("Тест работы с файлами...")
    
    try:
        # Создаем тестовую матрицу
        test_matrix = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
        
        # Записываем в файл
        write_matrix_to_file(test_matrix, "test_matrix_safe.txt")
        print("✓ Матрица записана в файл")
        
        # Читаем из файла
        loaded_matrix = read_matrix_from_file("test_matrix_safe.txt")
        print(f"✓ Матрица прочитана из файла: {loaded_matrix}")
        
        # Пробуем прочитать несуществующий файл
        try:
            read_matrix_from_file("non_existent_file.txt")
        except FileNotFoundError as e:
            print(f"✓ Корректно обработано отсутствие файла: {e}")
        
    except Exception as e:
        print(f"✗ Ошибка при работе с файлами: {e}")
        from exception_handler import save_error_report
        save_error_report(e, {'test': 'file_operations'})