print("Программа для сложения двух чисел")
print("Для выхода введите 'выход' или нажмите Ctrl+C\n")

while True:
    try:
        # Ввод первого числа
        input1 = input("Введите первое число (или 'выход' для завершения): ")
        if input1.lower() in ['выход', 'exit', 'quit', 'q']:
            print("Программа завершена.")
            break
        
        num1 = float(input1)
        
        # Ввод второго числа
        input2 = input("Введите второе число: ")
        if input2.lower() in ['выход', 'exit', 'quit', 'q']:
            print("Программа завершена.")
            break
            
        num2 = float(input2)
        
        # Вычисление и вывод суммы
        result = num1 + num2
        print(f"Результат: {num1} + {num2} = {result}")
        print("=" * 50)
        
    except ValueError:
        print("Ошибка! Пожалуйста, вводите только числа.")
        print("=" * 50)
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем.")
        break