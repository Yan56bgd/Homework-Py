def infinite_sum():
    """Бесконечный цикл для сложения двух чисел"""
    print("=== БЕСКОНЕЧНЫЙ КАЛЬКУЛЯТОР СУММЫ ===")
    print("Для выхода нажмите Ctrl+C\n")
    
    try:
        while True:
            try:
                num1 = float(input("Введите первое число: "))
                num2 = float(input("Введите второе число: "))
                result = num1 + num2
                print(f"Сумма: {num1} + {num2} = {result}\n")
                print("-" * 30)
            except ValueError:
                print("Ошибка: введите корректные числа!\n")
    except KeyboardInterrupt:
        print("\n Программа завершена!")

# infinite_sum()