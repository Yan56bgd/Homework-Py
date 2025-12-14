# HM 4.2.1.py
import cv2
import numpy as np
import argparse
import sys
import os
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def read_image(filename: str) -> Optional[np.ndarray]:
    """Чтение изображения на основе расширения файла"""
    if not os.path.exists(filename):
        print(f"Ошибка: файл {filename} не найден")
        return None
    
    # Определяем расширение файла
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            # Чтение изображения OpenCV
            img = cv2.imread(filename)
            if img is None:
                print(f"Ошибка: не удалось прочитать изображение {filename}")
                return None
            return img
        elif ext in ['.npy']:
            # Чтение numpy массива
            return np.load(filename)
        else:
            print(f"Ошибка: неподдерживаемый формат файла {ext}")
            return None
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None


def write_image(image: np.ndarray, filename: str) -> bool:
    """Запись изображения с учетом расширения"""
    if image is None:
        return False
    
    try:
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            cv2.imwrite(filename, image)
        elif ext in ['.npy']:
            np.save(filename, image)
        else:
            # По умолчанию сохраняем как PNG
            cv2.imwrite(filename, image)
        
        print(f"Изображение сохранено как {filename}")
        return True
    except Exception as e:
        print(f"Ошибка при сохранении файла {filename}: {e}")
        return False


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Эквализация гистограммы"""
    if len(image.shape) == 2:
        # Монохромное изображение
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:
        # Цветное изображение - преобразуем в YUV и эквализируем только Y канал
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return image


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Гамма-коррекция изображения"""
    # Нормализуем изображение
    image_normalized = image.astype(np.float32) / 255.0
    
    # Применяем гамма-коррекцию
    corrected = np.power(image_normalized, gamma)
    
    # Возвращаем к исходному диапазону
    corrected = (corrected * 255).astype(np.uint8)
    
    return corrected


def calculate_image_histogram(image: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Расчет гистограммы изображения"""
    if len(image.shape) == 2:
        # Монохромное изображение
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        return hist.flatten(), np.arange(bins)
    elif len(image.shape) == 3:
        # Цветное изображение
        hists = []
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hists.append(hist.flatten())
        
        # Возвращаем гистограммы для каждого канала
        return np.array(hists), np.arange(bins)
    else:
        return np.array([]), np.array([])


def plot_histogram(hist: np.ndarray, bins: np.ndarray, title: str = "Гистограмма"):
    """Построение графика гистограммы"""
    plt.figure(figsize=(10, 6))
    
    if len(hist.shape) == 1:
        # Монохромная гистограмма
        plt.bar(bins, hist, width=1.0, color='gray', alpha=0.7)
        plt.title(f"{title} (монохромная)")
    elif len(hist.shape) == 2 and hist.shape[0] == 3:
        # Цветная гистограмма
        colors = ('b', 'g', 'r')
        labels = ('Синий', 'Зеленый', 'Красный')
        for i, color in enumerate(colors):
            plt.plot(bins, hist[i], color=color, label=labels[i], alpha=0.7)
        plt.title(f"{title} (цветная)")
        plt.legend()
    
    plt.xlabel('Интенсивность')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем гистограмму как изображение
    plt.savefig('histogram.png')
    print("Гистограмма сохранена как 'histogram.png'")
    plt.show()


def process_image(args):
    """Основная функция обработки изображения"""
    # Чтение изображения
    image = read_image(args.input)
    if image is None:
        return
    
    print(f"Изображение прочитано: {image.shape}")
    
    # Выбор операции
    if args.operation == 'histogram':
        # Расчет и отображение гистограммы
        hist, bins = calculate_image_histogram(image, args.bins)
        plot_histogram(hist, bins, f"Гистограмма {args.input}")
        
    elif args.operation == 'equalize':
        # Эквализация гистограммы
        result = histogram_equalization(image)
        output_file = f"equalized_{os.path.basename(args.input)}"
        write_image(result, output_file)
        
        # Показываем гистограммы до и после
        hist_before, bins = calculate_image_histogram(image)
        hist_after, _ = calculate_image_histogram(result)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Исходное изображение')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        if len(hist_before.shape) == 1:
            plt.bar(bins, hist_before, width=1.0, color='gray', alpha=0.7)
        plt.title('Гистограмма до')
        
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('После эквализации')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        if len(hist_after.shape) == 1:
            plt.bar(bins, hist_after, width=1.0, color='gray', alpha=0.7)
        plt.title('Гистограмма после')
        
        plt.tight_layout()
        plt.savefig('equalization_comparison.png')
        plt.show()
        
    elif args.operation == 'gamma':
        # Гамма-коррекция
        result = gamma_correction(image, args.gamma)
        output_file = f"gamma_{args.gamma}_{os.path.basename(args.input)}"
        write_image(result, output_file)
        
        # Показываем результат
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Исходное (gamma=1.0)')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Гамма-коррекция (gamma={args.gamma})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('gamma_comparison.png')
        plt.show()
    
    else:
        print(f"Неизвестная операция: {args.operation}")


def main():
    parser = argparse.ArgumentParser(description='Обработка изображений')
    parser.add_argument('input', help='Входное изображение')
    parser.add_argument('operation', choices=['histogram', 'equalize', 'gamma'],
                       help='Операция: histogram - гистограмма, equalize - эквализация, gamma - гамма-коррекция')
    parser.add_argument('--bins', type=int, default=256,
                       help='Количество бинов для гистограммы (по умолчанию: 256)')
    parser.add_argument('--gamma', type=float, default=2.2,
                       help='Значение gamma для коррекции (по умолчанию: 2.2)')
    parser.add_argument('--output', help='Выходной файл (если не указан, генерируется автоматически)')
    
    args = parser.parse_args()
    process_image(args)


if __name__ == "__main__":
    main()