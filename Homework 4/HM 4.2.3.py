# wave_processing.py
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def read_wav_file(filename: str):
    """Чтение WAV файла"""
    try:
        with wave.open(filename, 'rb') as wav_file:
            # Получаем параметры файла
            params = wav_file.getparams()
            n_channels = params.nchannels
            sample_width = params.sampwidth
            framerate = params.framerate
            n_frames = params.nframes
            
            print(f"Параметры WAV файла:")
            print(f"  Каналов: {n_channels}")
            print(f"  Ширина сэмпла: {sample_width} байт")
            print(f"  Частота дискретизации: {framerate} Гц")
            print(f"  Количество кадров: {n_frames}")
            
            # Читаем все кадры
            frames = wav_file.readframes(n_frames)
            
            # Преобразуем в массив чисел в зависимости от ширины сэмпла
            if sample_width == 1:
                # 8-битные сэмплы (беззнаковые)
                data = np.frombuffer(frames, dtype=np.uint8)
                # Преобразуем в знаковые
                data = data.astype(np.int16) - 128
            elif sample_width == 2:
                # 16-битные сэмплы
                data = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 3:
                # 24-битные сэмплы (требуется специальная обработка)
                data = []
                for i in range(0, len(frames), 3):
                    # Объединяем 3 байта в 32-битное значение
                    value = struct.unpack('<i', frames[i:i+3] + b'\x00'[0:1])[0]
                    data.append(value >> 8)  # Сдвигаем, чтобы получить 24-битное значение
                data = np.array(data, dtype=np.int32)
            elif sample_width == 4:
                # 32-битные сэмплы
                data = np.frombuffer(frames, dtype=np.int32)
            else:
                raise ValueError(f"Неподдерживаемая ширина сэмпла: {sample_width}")
            
            # Если многоканальный, изменяем форму массива
            if n_channels > 1:
                data = data.reshape(-1, n_channels)
            
            return data, params
            
    except Exception as e:
        print(f"Ошибка при чтении WAV файла: {e}")
        return None, None


def write_wav_file(data, params, filename: str, quantize_bits: int = 16):
    """Запись WAV файла с квантованием"""
    try:
        n_channels = params.nchannels
        framerate = params.framerate
        
        # Квантование данных
        if quantize_bits == 8:
            # 8-битное квантование
            # Масштабируем до диапазона 0-255
            data_min = data.min()
            data_max = data.max()
            data_range = data_max - data_min
            
            if data_range > 0:
                quantized = ((data - data_min) / data_range * 255).astype(np.uint8)
            else:
                quantized = np.full_like(data, 128, dtype=np.uint8)
            
            sample_width = 1
            dtype = np.uint8
            
        elif quantize_bits == 16:
            # 16-битное квантование
            data_min = data.min()
            data_max = data.max()
            data_range = data_max - data_min
            
            if data_range > 0:
                # Масштабируем до диапазона -32768..32767
                quantized = ((data - data_min) / data_range * 65535 - 32768).astype(np.int16)
            else:
                quantized = np.zeros_like(data, dtype=np.int16)
            
            sample_width = 2
            dtype = np.int16
            
        elif quantize_bits == 24:
            # 24-битное квантование
            data_min = data.min()
            data_max = data.max()
            data_range = data_max - data_min
            
            if data_range > 0:
                # Масштабируем до диапазона -8388608..8388607
                quantized = ((data - data_min) / data_range * 16777215 - 8388608).astype(np.int32)
            else:
                quantized = np.zeros_like(data, dtype=np.int32)
            
            sample_width = 3
            dtype = np.int32
            
        else:
            raise ValueError(f"Неподдерживаемая битность квантования: {quantize_bits}")
        
        # Создаем новый WAV файл
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(framerate)
            
            # Преобразуем данные в байты
            if sample_width == 3:
                # Специальная обработка для 24-битных данных
                byte_data = bytearray()
                for sample in quantized.flatten():
                    # Преобразуем 32-битное значение в 3 байта
                    byte_data.extend(struct.pack('<i', sample)[:3])
                wav_file.writeframes(byte_data)
            else:
                wav_file.writeframes(quantized.tobytes())
        
        print(f"Файл сохранен как {filename} ({quantize_bits}-битное квантование)")
        
    except Exception as e:
        print(f"Ошибка при записи WAV файла: {e}")


def calculate_audio_histogram(data, bins: int = 256):
    """Расчет гистограммы аудиоданных"""
    # Преобразуем в одномерный массив
    if len(data.shape) > 1:
        data_flat = data.flatten()
    else:
        data_flat = data
    
    # Вычисляем гистограмму
    hist, bin_edges = np.histogram(data_flat, bins=bins, density=True)
    
    return hist, bin_edges


def plot_audio_histogram(hist, bin_edges, title: str = "Гистограмма аудио"):
    """Построение графика гистограммы аудио"""
    plt.figure(figsize=(10, 6))
    
    # Вычисляем центры бинов
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], 
            color='blue', alpha=0.7, edgecolor='black')
    
    plt.title(title)
    plt.xlabel('Амплитуда')
    plt.ylabel('Плотность вероятности')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем гистограмму
    plt.savefig('audio_histogram.png')
    print("Гистограмма аудио сохранена как 'audio_histogram.png'")
    plt.show()


def plot_waveform(data, framerate: int, title: str = "Волновая форма"):
    """Построение графика волновой формы"""
    plt.figure(figsize=(12, 6))
    
    # Создаем временную ось
    time = np.arange(len(data)) / framerate
    
    plt.plot(time, data, color='blue', alpha=0.7, linewidth=0.5)
    plt.title(title)
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('waveform.png')
    print("Волновая форма сохранена как 'waveform.png'")
    plt.show()


def process_wav_file(input_file: str, output_file: str = None, 
                    bins: int = 256, quantize_bits: int = 16):
    """Обработка WAV файла"""
    # Чтение WAV файла
    data, params = read_wav_file(input_file)
    
    if data is None:
        return
    
    # Преобразуем в одномерный массив для анализа
    if len(data.shape) > 1:
        data_mono = data.mean(axis=1)
    else:
        data_mono = data
    
    print(f"\nАнализ аудиоданных:")
    print(f"  Размер данных: {data_mono.shape}")
    print(f"  Минимальное значение: {data_mono.min():.2f}")
    print(f"  Максимальное значение: {data_mono.max():.2f}")
    print(f"  Среднее значение: {data_mono.mean():.2f}")
    
    # 1. Строим гистограмму
    hist, bin_edges = calculate_audio_histogram(data_mono, bins)
    plot_audio_histogram(hist, bin_edges, f"Гистограмма {input_file}")
    
    # 2. Строим волновую форму
    plot_waveform(data_mono[:min(10000, len(data_mono))], 
                  params.framerate, f"Волновая форма {input_file}")
    
    # 3. Квантование и сохранение
    if output_file is None:
        # Генерируем имя файла
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_quantized_{quantize_bits}bit.wav"
    
    write_wav_file(data, params, output_file, quantize_bits)
    
    print(f"\nОбработка завершена!")
    print(f"  Исходный файл: {input_file}")
    print(f"  Результат: {output_file}")
    print(f"  Битность квантования: {quantize_bits} бит")


if __name__ == "__main__":
    # Пример использования
    if len(sys.argv) > 1:
        process_wav_file(sys.argv[1])
    else:
        print("Использование: python wave_processing.py <input.wav> [output.wav]")