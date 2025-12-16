import math
from functools import partial
from typing import List, Tuple, Generator, Callable, Any

# ==================== Итераторы обхода изображения ====================

def linear_traversal(image: List[List[Any]]) -> Generator[Tuple[int, int, Any], None, None]:
    """Линейный построчный обход изображения."""
    h, w = len(image), len(image[0])
    for i in range(h):
        for j in range(w):
            yield (i, j, image[i][j])


def spiral_traversal(image: List[List[Any]]) -> Generator[Tuple[int, int, Any], None, None]:
    """Обход изображения по спирали из центра."""
    h, w = len(image), len(image[0])
    visited = [[False] * w for _ in range(h)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    dir_idx = 0
    ci, cj = h // 2, w // 2  # center
    visited[ci][cj] = True
    yield (ci, cj, image[ci][cj])
    total_cells = h * w
    visited_count = 1

    while visited_count < total_cells:
        dr, dc = directions[dir_idx]
        ni, nj = ci + dr, cj + dc
        if 0 <= ni < h and 0 <= nj < w and not visited[ni][nj]:
            visited[ni][nj] = True
            yield (ni, nj, image[ni][nj])
            ci, cj = ni, nj
            visited_count += 1
        else:
            dir_idx = (dir_idx + 1) % 4


def zigzag_traversal(image: List[List[Any]]) -> Generator[Tuple[int, int, Any], None, None]:
    """Зигзаг-обход изображения по диагоналям."""
    h, w = len(image), len(image[0])
    for d in range(h + w - 1):
        if d % 2 == 0:  # even diagonal: bottom to top
            i = min(d, h - 1)
            j = d - i
            while i >= 0 and j < w:
                yield (i, j, image[i][j])
                i -= 1
                j += 1
        else:  # odd diagonal: top to bottom
            j = min(d, w - 1)
            i = d - j
            while j >= 0 and i < h:
                yield (i, j, image[i][j])
                i += 1
                j -= 1


def peano_traversal(image: List[List[Any]]) -> Generator[Tuple[int, int, Any], None, None]:
    """Развертка Пеано для квадратных изображений размера 3^n."""
    h = len(image)
    w = len(image[0])
    
    # Проверяем, является ли изображение квадратным и степенью 3
    if h != w:
        yield from linear_traversal(image)
        return
    
    # Проверяем, является ли размер степенью 3
    size = h
    n = 0
    temp = size
    while temp % 3 == 0 and temp > 1:
        temp //= 3
        n += 1
    if temp != 1:
        yield from linear_traversal(image)
        return
    
    # Рекурсивная функция для генерации кривой Пеано
    def peano_curve_rec(x: int, y: int, size: int, n: int, direction: int) -> Generator[Tuple[int, int], None, None]:
        if n == 0:
            yield (x, y)
        else:
            sub_size = size // 3
            if direction == 0:  # normal direction
                order = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
                next_dir = [0, 1, 0, 0, 1, 1, 1, 0, 1]
            else:  # reversed direction
                order = [(0, 2), (0, 1), (0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 1), (2, 0)]
                next_dir = [1, 0, 1, 1, 0, 0, 0, 1, 0]
            
            for (i, j), nd in zip(order, next_dir):
                new_x = x + i * sub_size
                new_y = y + j * sub_size
                yield from peano_curve_rec(new_x, new_y, sub_size, n - 1, nd)
    
    for i, j in peano_curve_rec(0, 0, size, n, 0):
        yield (i, j, image[i][j])


# ==================== Конвертация в оттенки серого ====================

def to_grayscale(image: List[List[Tuple[int, int, int]]]) -> List[List[int]]:
    """Конвертация цветного изображения в оттенки серого с использованием лямбда-функций."""
    return [
        list(map(lambda pixel: int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]), row))
        for row in image
    ]


# ==================== Сверточный оператор ====================

def apply_convolution(
    image: List[List[float]],
    kernel: List[List[float]],
    traversal_func: Callable[[List[List[float]]], Generator[Tuple[int, int, float], None, None]]
) -> List[List[float]]:
    """
    Применяет свертку к изображению с использованием заданного ядра и метода обхода.
    Обрабатываются только внутренние пиксели, чтобы не выходить за границы.
    """
    h, w = len(image), len(image[0])
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    
    # Создаем результирующее изображение для внутренней области
    result_h, result_w = h - 2 * pad_h, w - 2 * pad_w
    result = [[0.0] * result_w for _ in range(result_h)]
    
    # Применяем свертку, используя заданный метод обхода
    for i, j, _ in traversal_func(image):
        if pad_h <= i < h - pad_h and pad_w <= j < w - pad_w:
            val = 0.0
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    val += image[i - pad_h + ki][j - pad_w + kj] * kernel[ki][kj]
            result[i - pad_h][j - pad_w] = val
    
    return result


def make_filter(
    kernel: List[List[float]],
    traversal_func: Callable[[List[List[float]]], Generator[Tuple[int, int, float], None, None]]
) -> Callable[[List[List[float]]], List[List[float]]]:
    """
    Создает фильтр с фиксированным ядром и методом обхода.
    Используется partial для фиксации параметров.
    """
    return partial(apply_convolution, kernel=kernel, traversal_func=traversal_func)


# Примеры ядер фильтров
AVERAGE_KERNEL_3x3 = [[1/9] * 3 for _ in range(3)]
GAUSSIAN_KERNEL_3x3 = [
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
]

SOBEL_X_KERNEL = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]


# ==================== Тестирование и метрики ====================

def mse(image1: List[List[float]], image2: List[List[float]]) -> float:
    """Вычисляет среднеквадратичную ошибку между двумя изображениями."""
    h, w = len(image1), len(image1[0])
    total = 0.0
    for i in range(h):
        for j in range(w):
            total += (image1[i][j] - image2[i][j]) ** 2
    return total / (h * w)


def test_filters(
    image: List[List[float]],
    kernel: List[List[float]],
    traversal_funcs: List[Tuple[str, Callable]]
) -> None:
    """
    Тестирует применение одного фильтра с разными методами обхода.
    Сравнивает результаты с помощью MSE.
    """
    results = []
    print(f"Тестирование фильтра с ядром {len(kernel)}x{len(kernel[0])}")
    print("-" * 50)
    
    for name, traversal_func in traversal_funcs:
        filter_func = make_filter(kernel, traversal_func)
        result = filter_func(image)
        results.append((name, result))
        print(f"{name}: размер результата {len(result)}x{len(result[0])}")
    
    # Сравниваем все результаты между собой
    print("\nСравнение результатов (MSE):")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            mse_value = mse(results[i][1], results[j][1])
            print(f"{results[i][0]} vs {results[j][0]}: MSE = {mse_value:.10f}")
            if mse_value < 1e-10:
                print("  ✓ Результаты идентичны")
            else:
                print("  ✗ Результаты различаются")
    print()


# ==================== Пример использования ====================

def main():
    """Пример использования всех реализованных функций."""
    
    # 1. Создаем тестовое цветное изображение 9x9 (степень 3 для Пеано)
    color_image = [
        [(i * 28, j * 28, (i + j) * 14) for j in range(9)] 
        for i in range(9)
    ]
    
    print("1. Исходное цветное изображение 9x9")
    print("   Размер:", len(color_image), "x", len(color_image[0]))
    
    # 2. Конвертируем в оттенки серого
    gray_image = to_grayscale(color_image)
    print("\n2. Изображение в оттенках серого")
    print("   Размер:", len(gray_image), "x", len(gray_image[0]))
    print("   Пример пикселя (0,0):", gray_image[0][0])
    
    # 3. Тестируем разные фильтры с разными методами обхода
    traversal_methods = [
        ("Линейный", linear_traversal),
        ("Спираль", spiral_traversal),
        ("Зигзаг", zigzag_traversal),
        ("Пеано", peano_traversal)
    ]
    
    print("\n3. Тестирование усредняющего фильтра 3x3")
    test_filters(gray_image, AVERAGE_KERNEL_3x3, traversal_methods)
    
    print("\n4. Тестирование фильтра Гаусса 3x3")
    test_filters(gray_image, GAUSSIAN_KERNEL_3x3, traversal_methods)
    
    # 4. Демонстрация фильтра Собеля
    print("\n5. Демонстрация фильтра Собеля (обнаружение границ)")
    sobel_filter = make_filter(SOBEL_X_KERNEL, linear_traversal)
    sobel_result = sobel_filter(gray_image)
    print(f"   Размер результата: {len(sobel_result)}x{len(sobel_result[0])}")
    print(f"   Минимальное значение: {min(min(row) for row in sobel_result):.2f}")
    print(f"   Максимальное значение: {max(max(row) for row in sobel_result):.2f}")
    
    # 5. Тестирование на изображении другого размера (не степень 3)
    print("\n6. Тестирование на изображении 8x8 (не степень 3)")
    small_image = [[i + j for j in range(8)] for i in range(8)]
    
    # Для нестепени 3 метод Пеано автоматически переключится на линейный обход
    small_traversal_methods = [
        ("Линейный", linear_traversal),
        ("Спираль", spiral_traversal),
        ("Зигзаг", zigzag_traversal),
        ("Пеано (автопереключение)", peano_traversal)
    ]
    
    test_filters(small_image, AVERAGE_KERNEL_3x3, small_traversal_methods)


if __name__ == "__main__":
    main()