# HM 4.2.2.py
import argparse
import image_processing as ip
import wave_processing as wp
import sys
import os


def main():
    parser = argparse.ArgumentParser(description='Обработка изображений и аудио')
    subparsers = parser.add_subparsers(dest='command', help='Команда')
    
    # Парсер для обработки изображений
    img_parser = subparsers.add_parser('image', help='Обработка изображений')
    img_parser.add_argument('input', help='Входное изображение')
    img_parser.add_argument('operation', choices=['histogram', 'equalize', 'gamma'],
                           help='Операция: histogram - гистограмма, equalize - эквализация, gamma - гамма-коррекция')
    img_parser.add_argument('--bins', type=int, default=256,
                           help='Количество бинов для гистограммы')
    img_parser.add_argument('--gamma', type=float, default=2.2,
                           help='Значение gamma для коррекции')
    
    # Парсер для обработки аудио
    audio_parser = subparsers.add_parser('audio', help='Обработка аудио')
    audio_parser.add_argument('input', help='Входной WAV файл')
    audio_parser.add_argument('--bins', type=int, default=256,
                             help='Количество бинов для гистограммы')
    audio_parser.add_argument('--quantize', type=int, default=8,
                             help='Битность квантования (8, 16, 24)')
    audio_parser.add_argument('--output', help='Выходной файл')
    
    args = parser.parse_args()
    
    if args.command == 'image':
        # Обработка изображения
        ip.process_image(args)
    elif args.command == 'audio':
        # Обработка аудио
        wp.process_wav_file(args.input, args.output, args.bins, args.quantize)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()