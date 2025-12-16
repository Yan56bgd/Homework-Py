# ========== Hist.py ==========
import cv2
import numpy as np

class HistogramAnalyzer:
    """Класс для анализа гистограмм с дополнительной функциональностью"""
    
    @staticmethod
    def normalize_histogram(histogram):
        """Нормализация гистограммы"""
        return histogram / (histogram.sum() + 1e-7)
    
    @staticmethod
    def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
        """Сравнение двух гистограмм"""
        hist1_norm = HistogramAnalyzer.normalize_histogram(hist1).astype(np.float32)
        hist2_norm = HistogramAnalyzer.normalize_histogram(hist2).astype(np.float32)
        return cv2.compareHist(hist1_norm, hist2_norm, method)
    
    @staticmethod
    def equalize_image(image):
        """Эквализация гистограммы изображения"""
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            # Для цветных изображений - эквализация каждого канала
            channels = cv2.split(image)
            eq_channels = []
            for ch in channels:
                eq_channels.append(cv2.equalizeHist(ch))
            return cv2.merge(eq_channels)

# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========
if __name__ == "__main__":
    from encoder import HistogramManager, JSONHistogramStrategy, BinaryHistogramStrategy
    
    # Создаем менеджер гистограмм
    manager = HistogramManager()
    
    # Вычисляем гистограмму
    histogram = manager.compute_histogram('test_image.jpg', processor_type='color')
    
    # Сохраняем в JSON формате
    manager.save_histogram(histogram, 'histogram.json')
    
    # Меняем стратегию на бинарную
    manager.set_strategy(BinaryHistogramStrategy())
    manager.save_histogram(histogram, 'histogram.bin')
    
    print("Гистограммы сохранены в разных форматах!")