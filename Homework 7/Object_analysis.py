import cv2
import numpy as np
from abc import ABC, abstractmethod
from functools import wraps

# ========== ШАБЛОННЫЙ МЕТОД ==========
class ImageProcessor(ABC):
    """Абстрактный класс с шаблонным методом обработки изображений"""
    
    def process(self, image_path):
        """Шаблонный метод обработки изображения"""
        image = self._load_image(image_path)
        processed_image = self._preprocess(image)
        objects = self._extract_objects(processed_image)
        params = self._compute_object_params(objects, processed_image)
        return processed_image, objects, params
    
    def _load_image(self, image_path):
        """Загрузка изображения"""
        return cv2.imread(image_path)
    
    @abstractmethod
    def _preprocess(self, image):
        """Предобработка изображения"""
        pass
    
    @abstractmethod
    def _extract_objects(self, image):
        """Извлечение объектов"""
        pass
    
    def _compute_object_params(self, objects, image):
        """Вычисление параметров объектов (по умолчанию - моменты)"""
        params = []
        for obj in objects:
            if len(obj) > 0:
                moments = cv2.moments(obj)
                hu_moments = cv2.HuMoments(moments)
                params.append({
                    'moments': moments,
                    'hu_moments': hu_moments.flatten()
                })
        return params

class MonoProcessor(ImageProcessor):
    """Обработчик монохромных изображений"""
    
    def _preprocess(self, image):
        """Предобработка: преобразование в монохром и фильтр Гаусса"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    def _extract_objects(self, image):
        """Извлечение объектов с помощью детектора границ Canny"""
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

class ColorProcessor(ImageProcessor):
    """Обработчик цветных изображений"""
    
    def _preprocess(self, image):
        """Предобработка: сегментация с distance transform и watershed"""
        if len(image.shape) == 2:
            # Если изображение уже монохромное
            gray = image.copy()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Удаление шума
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Distance transform и watershed
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(opening, sure_fg)
        
        # Маркировка компонентов
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        return markers
    
    def _extract_objects(self, image):
        """Извлечение объектов из маркеров watershed"""
        # Преобразуем маркеры в контуры
        markers_uint8 = np.uint8(image)
        contours, _ = cv2.findContours(markers_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

# ========== ДЕКОРАТОР ==========
def hu_moments_decorator(processor_class):
    """Декоратор для добавления вычисления моментов Hu к процессору"""
    
    class DecoratedProcessor(processor_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def _compute_object_params(self, objects, image):
            """Переопределенный метод с вычислением моментов Hu"""
            params = super()._compute_object_params(objects, image)
            
            # Добавляем дополнительные вычисления моментов Hu
            for i, obj in enumerate(objects):
                if len(obj) > 0:
                    # Вычисляем моменты Hu для более точного анализа
                    moments = cv2.moments(obj)
                    hu_moments = cv2.HuMoments(moments)
                    
                    if i < len(params):
                        params[i]['hu_moments_detailed'] = hu_moments.flatten()
                        params[i]['area'] = cv2.contourArea(obj)
                        params[i]['perimeter'] = cv2.arcLength(obj, True)
            
            return params
    
    return DecoratedProcessor

# ========== ФАБРИКА ПРОЦЕССОРОВ ==========
class ProcessorFactory:
    """Фабрика для создания процессоров с декораторами"""
    
    @staticmethod
    def create_processor(processor_type, use_decorator=False):
        processors = {
            'mono': MonoProcessor,
            'color': ColorProcessor
        }
        
        if processor_type not in processors:
            raise ValueError(f"Неизвестный тип процессора: {processor_type}")
        
        processor_class = processors[processor_type]
        
        if use_decorator:
            processor_class = hu_moments_decorator(processor_class)
        
        return processor_class()

# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========
if __name__ == "__main__":
    # Пример использования
    factory = ProcessorFactory()
    
    # Создаем процессор для монохромных изображений с декоратором
    mono_processor = factory.create_processor('mono', use_decorator=True)
    
    # Создаем процессор для цветных изображений без декоратора
    color_processor = factory.create_processor('color', use_decorator=False)
    
    print("Процессоры созданы успешно!")