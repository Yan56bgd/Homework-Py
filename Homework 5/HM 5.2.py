# images.py
from abc import ABC, abstractmethod
import numpy as np


class BaseImage(ABC):
    def __init__(self, data: np.ndarray):
        self.data = data

    @abstractmethod
    def __repr__(self):
        pass


class BinaryImage(BaseImage):
    def __repr__(self):
        return f"BinaryImage(shape={self.data.shape})"


class MonochromeImage(BaseImage):
    def __repr__(self):
        return f"MonochromeImage(shape={self.data.shape})"


class ColorImage(BaseImage):
    def __repr__(self):
        return f"ColorImage(shape={self.data.shape})"


class ImageConverter:

    @staticmethod
    def mono_to_mono(img: MonochromeImage) -> MonochromeImage:
        data = (img.data - img.data.min()) / (img.data.ptp() + 1e-9) * 255
        return MonochromeImage(data.astype(np.uint8))

    @staticmethod
    def color_to_color(img: ColorImage) -> ColorImage:
        data = img.data.astype(float)
        for i in range(3):
            channel = data[:, :, i]
            data[:, :, i] = (channel - channel.min()) / (channel.ptp() + 1e-9) * 255
        return ColorImage(data.astype(np.uint8))

    @staticmethod
    def binary_to_binary(img: BinaryImage) -> BinaryImage:
        return BinaryImage(img.data.copy())

    @staticmethod
    def color_to_mono(img: ColorImage) -> MonochromeImage:
        gray = img.data.mean(axis=2)
        return MonochromeImage(gray.astype(np.uint8))

    @staticmethod
    def mono_to_color(img: MonochromeImage, palette: list[tuple[int, int, int]]) -> ColorImage:
        h, w = img.data.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        for shade, color in enumerate(palette):
            result[img.data == shade] = color
        return ColorImage(result)

    @staticmethod
    def mono_to_binary(img: MonochromeImage, threshold=128) -> BinaryImage:
        return BinaryImage((img.data > threshold).astype(np.uint8))

    @staticmethod
    def binary_to_mono(img: BinaryImage) -> MonochromeImage:
        dist = np.zeros_like(img.data, dtype=float)
        h, w = img.data.shape
        for i in range(h):
            for j in range(w):
                if img.data[i, j] == 1:
                    dist[i, j] = 0
                else:
                    ones = np.argwhere(img.data == 1)
                    dist[i, j] = np.min(np.linalg.norm(ones - np.array([i, j]), axis=1)) if ones.size else 255
        dist = dist / dist.max() * 255
        return MonochromeImage(dist.astype(np.uint8))

    @staticmethod
    def color_to_binary(img: ColorImage) -> BinaryImage:
        return ImageConverter.mono_to_binary(ImageConverter.color_to_mono(img))

    @staticmethod
    def binary_to_color(img: BinaryImage, palette) -> ColorImage:
        return ImageConverter.mono_to_color(ImageConverter.binary_to_mono(img), palette)


if __name__ == "__main__":
    # создаём тестовые изображения
    mono = MonochromeImage(np.array([[10, 120], [200, 255]], dtype=np.uint8))
    binary = BinaryImage(np.array([[0, 1], [1, 0]], dtype=np.uint8))
    color = ColorImage(np.random.randint(0, 255, (2, 2, 3), dtype=np.uint8))

    print("Исходный моно:", mono)
    print("Исходный бинарный:", binary)
    print("Исходный цветной:", color)

    print("Моно → Моно:", ImageConverter.mono_to_mono(mono))
    print("Цвет → Моно:", ImageConverter.color_to_mono(color))
    print("Цвет → Цвет:", ImageConverter.color_to_color(color))
    print("Бинар → Моно:", ImageConverter.binary_to_mono(binary))
    print("Моно → Бинар:", ImageConverter.mono_to_binary(mono))
    print("Цвет → Бинар:", ImageConverter.color_to_binary(color))
