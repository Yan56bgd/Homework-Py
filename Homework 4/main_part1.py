# main_part1.py
import matrix_operations as mo
import random


def main():
    print("Часть 1: Базовые матрично-векторные операции")
    print("=" * 60)
    
    # Демонстрация всех функций
    print("\n1. Умножение матрица-матрица:")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = mo.matrix_multiply(A, B)
    print(f"   {A} * {B} = {C}")
    
    print("\n2. Умножение матрица-вектор:")
    matrix = [[1, 2, 3], [4, 5, 6]]
    vector = [7, 8, 9]
    result = mo.matrix_vector_multiply(matrix, vector)
    print(f"   {matrix} * {vector} = {result}")
    
    print("\n3. След матрицы:")
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    trace = mo.matrix_trace(matrix)
    print(f"   trace({matrix}) = {trace}")
    
    print("\n4. Скалярное произведение:")
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    dot = mo.dot_product(v1, v2)
    print(f"   {v1} · {v2} = {dot}")
    
    print("\n5. Гистограмма:")
    data = [random.uniform(0, 10) for _ in range(20)]
    hist, bins = mo.calculate_histogram(data, 5)
    print(f"   Данные: {data[:5]}...")
    print(f"   Границы бинов: {bins}")
    print(f"   Гистограмма: {hist}")
    
    print("\n6. Фильтрация ядерным фильтром:")
    data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    kernel = [-1, 0, 1]
    filtered = mo.kernel_filter(data, kernel)
    print(f"   Данные: {data}")
    print(f"   Ядро: {kernel}")
    print(f"   Результат: {filtered}")
    
    print("\n7. Чтение/запись файлов:")
    # Записываем тестовые данные
    test_data = [[1.1, 2.2], [3.3, 4.4]]
    mo.write_matrix_to_file(test_data, "test_data.txt")
    
    # Читаем обратно
    loaded = mo.read_matrix_from_file("test_data.txt")
    print(f"   Записано: {test_data}")
    print(f"   Прочитано: {loaded}")
    
    # Запускаем бенчмарки
    print("\n" + "=" * 60)
    print("Запуск бенчмарков для измерения времени...")
    mo.run_benchmark()


if __name__ == "__main__":
    main()