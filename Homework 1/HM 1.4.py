print("=== ПОИСК НАИБОЛЬШЕГО ЧИСЛА ===")
print("Вводите числа (больше 0)")
print("Для завершения введите 0")
print("-" * 30)
    
max_number = 0  # Начальное значение максимума
    
while True:
        try:
            number = float(input("Введите число: "))
            
            if number == 0:
                break  # Выход из цикла при вводе 0
            
            if number < 0:
                print("Число должно быть больше 0!")
                continue
            
            if number > max_number:
                max_number = number
                print(f"Новый максимум: {max_number}")
                
        except ValueError:
            print("Ошибка! Введите корректное число.")
    
    # Вывод результата
if max_number > 0:
        print(f"\nНаибольшее число среди введенных: {max_number}")
else:
        print("\nНе было введено ни одного положительного числа.")