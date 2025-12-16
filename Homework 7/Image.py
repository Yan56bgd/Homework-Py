from abc import ABC, abstractmethod
import cv2
import numpy as np
from PIL import Image as PILImage
import imageio

class ImageReader(ABC):
    """Абстрактный класс для чтения изображений"""
    
    @abstractmethod
    def read(self, filepath):
        pass
    
    @abstractmethod
    def get_format(self):
        pass

class OpenCVImageReader(ImageReader):
    """Чтение изображений с помощью OpenCV"""
    
    def read(self, filepath):
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {filepath}")
        return image
    
    def get_format(self):
        return "OpenCV"

class PILImageReader(ImageReader):
    """Чтение изображений с помощью PIL"""
    
    def read(self, filepath):
        image = PILImage.open(filepath)
        return np.array(image)
    
    def get_format(self):
        return "PIL"

class ImageIOReader(ImageReader):
    """Чтение изображений с помощью imageio"""
    
    def read(self, filepath):
        image = imageio.imread(filepath)
        return image
    
    def get_format(self):
        return "ImageIO"

class DICOMImageReader(ImageReader):
    """Чтение DICOM изображений (медицинские изображения)"""
    
    def read(self, filepath):
        try:
            import pydicom
            ds = pydicom.dcmread(filepath)
            image = ds.pixel_array
            return image
        except ImportError:
            raise ImportError("Для чтения DICOM требуется установить pydicom")
    
    def get_format(self):
        return "DICOM"

class ImageReaderFactory:
    """Абстрактная фабрика для создания ридеров изображений"""
    
    @staticmethod
    def create_reader(reader_type):
        readers = {
            'opencv': OpenCVImageReader,
            'pil': PILImageReader,
            'imageio': ImageIOReader,
            'dicom': DICOMImageReader
        }
        
        if reader_type not in readers:
            raise ValueError(f"Неизвестный тип ридера: {reader_type}")
        
        return readers[reader_type]()
    
    @staticmethod
    def get_reader_by_extension(filepath):
        """Создание ридера на основе расширения файла"""
        extension = filepath.lower().split('.')[-1]
        
        # Сопоставление расширений с ридерами
        extension_map = {
            'jpg': 'opencv',
            'jpeg': 'opencv',
            'png': 'opencv',
            'bmp': 'opencv',
            'tiff': 'pil',
            'tif': 'pil',
            'dcm': 'dicom',
            'dicom': 'dicom'
        }
        
        reader_type = extension_map.get(extension, 'opencv')
        return ImageReaderFactory.create_reader(reader_type)

class ImageProcessorFacade:
    """Фасад для обработки изображений со всей функциональностью"""
    
    def __init__(self):
        self.reader_factory = ImageReaderFactory()
        self.histogram_manager = None
        self.image_processor = None
    
    def load_image(self, filepath, reader_type=None):
        """Загрузка изображения с помощью фабрики ридеров"""
        if reader_type:
            reader = self.reader_factory.create_reader(reader_type)
        else:
            reader = self.reader_factory.get_reader_by_extension(filepath)
        
        print(f"Используется ридер: {reader.get_format()}")
        return reader.read(filepath)
    
    def process_image(self, image, processor_type='mono', use_decorator=True):
        """Обработка изображения с помощью процессора из Object_analysis.py"""
        from Object_analysis import ProcessorFactory
        
        # Создаем временный файл для передачи изображения процессору
        temp_path = 'temp_image.jpg'
        cv2.imwrite(temp_path, image)
        
        # Создаем и используем процессор
        self.image_processor = ProcessorFactory().create_processor(
            processor_type, 
            use_decorator
        )
        
        return self.image_processor.process(temp_path)
    
    def analyze_histogram(self, image, output_format='json'):
        """Анализ гистограммы изображения"""
        from encoder import HistogramManager, JSONHistogramStrategy, BinaryHistogramStrategy
        
        if self.histogram_manager is None:
            self.histogram_manager = HistogramManager()
        
        # Вычисляем гистограмму
        temp_path = 'temp_hist_image.jpg'
        cv2.imwrite(temp_path, image)
        
        processor_type = 'color' if len(image.shape) == 3 else 'mono'
        histogram = self.histogram_manager.compute_histogram(
            temp_path, 
            processor_type
        )
        
        # Выбираем стратегию сохранения
        if output_format == 'binary':
            self.histogram_manager.set_strategy(BinaryHistogramStrategy())
            output_file = 'histogram.bin'
        else:
            self.histogram_manager.set_strategy(JSONHistogramStrategy())
            output_file = 'histogram.json'
        
        # Сохраняем гистограмму
        self.histogram_manager.save_histogram(histogram, output_file)
        
        return histogram, output_file

# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========
if __name__ == "__main__":
    # Создаем фасад для обработки изображений
    facade = ImageProcessorFacade()
    
    # Загружаем изображение
    image = facade.load_image('test_image.jpg')
    print(f"Изображение загружено, размер: {image.shape}")
    
    # Обрабатываем изображение
    processed_image, objects, params = facade.process_image(
        image, 
        processor_type='color',
        use_decorator=True
    )
    print(f"Найдено объектов: {len(objects)}")
    
    # Анализируем гистограмму
    histogram, hist_file = facade.analyze_histogram(image, output_format='json')
    print(f"Гистограмма сохранена в файл: {hist_file}")
    
    print("Обработка завершена успешно!")