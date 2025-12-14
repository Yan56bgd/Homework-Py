# HM 6.1.py
"""
Модуль для обработки исключений в операциях с данными и файлами
"""
import traceback
import sys
from datetime import datetime
from typing import Any, Callable, Optional
import os


class DataValidationError(Exception):
    """Исключение для ошибок валидации данных"""
    pass


class FileOperationError(Exception):
    """Исключение для ошибок операций с файлами"""
    pass


def handle_file_operation(func: Callable) -> Callable:
    """
    Декоратор для обработки исключений при операциях с файлами
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise FileOperationError(f"Файл не найден: {e.filename}") from e
        except PermissionError as e:
            raise FileOperationError(f"Нет доступа к файлу: {e.filename}") from e
        except IsADirectoryError as e:
            raise FileOperationError(f"Путь является директорией, а не файлом: {e.filename}") from e
        except IOError as e:
            raise FileOperationError(f"Ошибка ввода-вывода: {str(e)}") from e
        except UnicodeDecodeError as e:
            raise FileOperationError(f"Ошибка кодировки файла: {str(e)}") from e
        except Exception as e:
            raise FileOperationError(f"Непредвиденная ошибка при работе с файлом: {str(e)}") from e
    
    return wrapper


def handle_data_operation(func: Callable) -> Callable:
    """
    Декоратор для обработки исключений при операциях с данными
    """
    def wrapper(*args, **kwargs):
        try:
            # Предварительная проверка аргументов
            if not args and not kwargs:
                return func(*args, **kwargs)
            
            # Проверка для матричных операций
            if func.__name__ in ['matrix_multiply', 'matrix_vector_multiply', 'matrix_trace']:
                if args and isinstance(args[0], list):
                    if not all(len(row) == len(args[0][0]) for row in args[0]):
                        raise DataValidationError("Несоответствие размеров строк матрицы")
            
            return func(*args, **kwargs)
        except IndexError as e:
            raise DataValidationError(f"Выход за границы массива: {str(e)}") from e
        except ValueError as e:
            raise DataValidationError(f"Неверное значение: {str(e)}") from e
        except TypeError as e:
            raise DataValidationError(f"Неверный тип данных: {str(e)}") from e
        except ZeroDivisionError as e:
            raise DataValidationError(f"Деление на ноль: {str(e)}") from e
        except MemoryError as e:
            raise DataValidationError(f"Недостаточно памяти: {str(e)}") from e
        except Exception as e:
            raise DataValidationError(f"Ошибка при обработке данных: {str(e)}") from e
    
    return wrapper


def validate_matrix(matrix: Any, min_rows: int = 1, min_cols: int = 1) -> bool:
    """
    Проверка корректности матрицы
    """
    if not isinstance(matrix, list):
        return False
    
    if not matrix or len(matrix) < min_rows:
        return False
    
    if not all(isinstance(row, list) for row in matrix):
        return False
    
    first_len = len(matrix[0])
    if first_len < min_cols:
        return False
    
    if not all(len(row) == first_len for row in matrix):
        return False
    
    # Проверка типов элементов
    for row in matrix:
        for element in row:
            if not isinstance(element, (int, float)):
                return False
    
    return True


def validate_vector(vector: Any, min_size: int = 1) -> bool:
    """
    Проверка корректности вектора
    """
    if not isinstance(vector, list):
        return False
    
    if len(vector) < min_size:
        return False
    
    # Проверка типов элементов
    for element in vector:
        if not isinstance(element, (int, float)):
            return False
    
    return True


def validate_file_path(filepath: str, check_exists: bool = True, 
                      check_writable: bool = False) -> bool:
    """
    Проверка корректности пути к файлу
    """
    if not isinstance(filepath, str) or not filepath:
        return False
    
    # Проверка существования файла
    if check_exists and not os.path.exists(filepath):
        return False
    
    # Проверка возможности записи
    if check_writable:
        try:
            # Проверяем, можем ли мы записать в файл
            dir_path = os.path.dirname(filepath) or '.'
            if not os.access(dir_path, os.W_OK):
                return False
        except Exception:
            return False
    
    return True


def safe_execute(func: Callable, *args, error_message: str = "", 
                default_return: Any = None, **kwargs) -> Any:
    """
    Безопасное выполнение функции с обработкой исключений
    """
    try:
        return func(*args, **kwargs)
    except FileOperationError as e:
        print(f"Ошибка файловой операции: {e}")
        if error_message:
            print(error_message)
        return default_return
    except DataValidationError as e:
        print(f"Ошибка валидации данных: {e}")
        if error_message:
            print(error_message)
        return default_return
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")
        print(f"Трассировка: {traceback.format_exc()}")
        if error_message:
            print(error_message)
        return default_return


def create_error_report(error: Exception, context: dict = None) -> str:
    """
    Создание отчета об ошибке
    """
    report = [
        f"=== ОТЧЕТ ОБ ОШИБКЕ ===",
        f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Тип ошибки: {type(error).__name__}",
        f"Сообщение: {str(error)}",
        f"Трассировка:",
        f"{traceback.format_exc()}",
    ]
    
    if context:
        report.append(f"Контекст:")
        for key, value in context.items():
            report.append(f"  {key}: {value}")
    
    return "\n".join(report)


def save_error_report(error: Exception, context: dict = None, 
                     filename: str = "error_report.txt"):
    """
    Сохранение отчета об ошибке в файл
    """
    report = create_error_report(error, context)
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(report + "\n\n")
        print(f"Отчет об ошибке сохранен в {filename}")
    except Exception as e:
        print(f"Не удалось сохранить отчет об ошибке: {e}")