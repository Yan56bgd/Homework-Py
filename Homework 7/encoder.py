# ========== encoder.py ==========
from abc import ABC, abstractmethod
import cv2
import numpy as np
import json
import pickle

class HistogramStrategy(ABC):
    """Абстрактный класс стратегии для работы с гистограммами"""
    
    @abstractmethod
    def read_histogram(self, filepath):
        pass
    
    @abstractmethod
    def write_histogram(self, histogram, filepath):
        pass

class JSONHistogramStrategy(HistogramStrategy):
    """Стратегия для работы с JSON форматом"""
    
    def read_histogram(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return np.array(data['histogram'])
    
    def write_histogram(self, histogram, filepath):
        data = {
            'histogram': histogram.tolist(),
            'shape': list(histogram.shape)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

class BinaryHistogramStrategy(HistogramStrategy):
    """Стратегия для работы с бинарным форматом (pickle)"""
    
    def read_histogram(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def write_histogram(self, histogram, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(histogram, f)

class CSVHistogramStrategy(HistogramStrategy):
    """Стратегия для работы с CSV форматом"""
    
    def read_histogram(self, filepath):
        return np.loadtxt(filepath, delimiter=',')
    
    def write_histogram(self, histogram, filepath):
        np.savetxt(filepath, histogram, delimiter=',')

class HistogramManager:
    """Менеджер гистограмм с поддержкой различных форматов"""
    
    def __init__(self, strategy=None):
        self.strategy = strategy or JSONHistogramStrategy()
        self.processor = None  # Ссылка на процессор из Object_analysis.py
    
    def set_strategy(self, strategy):
        """Установка стратегии работы с гистограммами"""
        self.strategy = strategy
    
    def compute_histogram(self, image_path, processor_type='mono'):
        """Вычисление гистограммы с использованием процессора из Object_analysis.py"""
        from Object_analysis import ProcessorFactory
        
        # Создаем процессор
        self.processor = ProcessorFactory().create_processor(processor_type)
        
        # Обрабатываем изображение
        processed_image, _, _ = self.processor.process(image_path)
        
        # Вычисляем гистограмму
        if len(processed_image.shape) == 2:
            # Монохромное изображение
            histogram = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
        else:
            # Цветное изображение
            hist_channels = []
            for i in range(processed_image.shape[2]):
                hist = cv2.calcHist([processed_image], [i], None, [256], [0, 256])
                hist_channels.append(hist)
            histogram = np.concatenate(hist_channels, axis=0)
        
        return histogram.flatten()
    
    def save_histogram(self, histogram, filepath):
        """Сохранение гистограммы в файл"""
        self.strategy.write_histogram(histogram, filepath)
    
    def load_histogram(self, filepath):
        """Загрузка гистограммы из файла"""
        return self.strategy.read_histogram(filepath)