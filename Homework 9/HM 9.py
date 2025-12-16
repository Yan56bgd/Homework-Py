from PIL import Image
import numpy as np
import re

# 1. Загружаем изображение
image_path = "Снимок экрана 2025-12-16 112836.png"
image = Image.open(image_path)

# Конвертируем в оттенки серого для удобства обработки
gray_image = image.convert("L")
gray_array = np.array(gray_image)

# 2. Разбиваем изображение на блоки 16x16
block_size = 16
height, width = gray_array.shape
# Обрезаем изображение до размера, кратного block_size
new_height = height - (height % block_size)
new_width = width - (width % block_size)
gray_array = gray_array[:new_height, :new_width]

# 3. Бинаризуем изображение (порог = 128)
threshold = 128
binary_array = np.where(gray_array > threshold, 1, 0)  # 1 - белый, 0 - черный

# 4. Преобразуем изображение: развертка по блокам 16x16
blocks_h = new_height // block_size
blocks_w = new_width // block_size

block_values = []
for i in range(blocks_h):
    row = []
    for j in range(blocks_w):
        block = binary_array[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
        # Определяем значение блока: если черных пикселей (0) больше половины -> 0, иначе 1
        black_count = np.sum(block == 0)
        total_pixels = block_size * block_size
        if black_count > total_pixels / 2:
            row.append(0)
        else:
            row.append(1)
    block_values.append(row)

# 5. Развертка исходного изображения по строкам (одномерный список значений блоков)
flat_block_values = [str(val) for row in block_values for val in row]

# 6. Преобразуем в строку вида 'r'00001001...00000''
result_string = 'r\'' + ''.join(flat_block_values) + '\''
print("Длина символьной строки (количество патчей):", len(flat_block_values))
print("Первые 100 символов строки:", result_string[:100] + "...")

# 7. Поиск паттернов "крест" с использованием регулярных выражений
# Паттерн "крест" в виде строки (по строкам, затем объединение)
cross_pattern = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])
pattern_string = ''.join(cross_pattern.flatten().astype(str))  # "010111010"

# Преобразуем бинарное изображение в строку для поиска (по строкам)
binary_string = ''.join(binary_array.flatten().astype(str))

# Используем регулярное выражение для поиска паттерна
# Учитываем, что паттерн 3x3, поэтому в строке это будет последовательность из 9 символов,
# но с учётом ширины изображения (переход на следующую строку)
height, width = binary_array.shape
# Создаем строку, где каждая строка изображения разделена символом перевода строки
rows = [''.join(binary_array[i].astype(str)) for i in range(height)]
text = '\n'.join(rows)

# Регулярное выражение для поиска паттерна "крест" в тексте с учётом структуры 3 строки по 3 символа
# Паттерн: первая строка: 0 1 0, вторая: 1 1 1, третья: 0 1 0
# В тексте строки разделены '\n', поэтому используем флаг re.DOTALL для работы с многострочным текстом
pattern = re.compile(r'0 1 0\n1 1 1\n0 1 0'.replace(' ', ''), re.DOTALL)

# Находим все совпадения
matches = pattern.findall(text)
print("Количество паттернов 'крест' в изображении:", len(matches))

# Дополнительно: визуализация для проверки (опционально)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(image)
axes[0, 0].set_title("Исходное изображение")
axes[0, 0].axis('off')

axes[0, 1].imshow(binary_array, cmap='gray')
axes[0, 1].set_title("Бинаризованное изображение")
axes[0, 1].axis('off')

# Показываем разбиение на блоки 16x16
blocked_image = np.zeros_like(binary_array)
for i in range(blocks_h):
    for j in range(blocks_w):
        blocked_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block_values[i][j]
axes[1, 0].imshow(blocked_image, cmap='gray')
axes[1, 0].set_title("Блоки 16x16 (0-черный, 1-белый)")
axes[1, 0].axis('off')

# Отмечаем найденные паттерны "крест" (для наглядности)
marked_image = np.array(binary_array)
for match in matches:
    # Для простоты отобразим только первый найденный паттерн
    # В реальности нужно отметить все, но это может быть нагромождением
    pass
axes[1, 1].imshow(marked_image, cmap='gray')
axes[1, 1].set_title("Бинаризованное с отмеченными паттернами")
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()