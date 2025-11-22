def find_max_number():
    """Поиск наибольшего числа среди введенных"""
    print("=== ПОИСК НАИБОЛЬШЕГО ЧИСЛА ===")
    print("Вводите числа (больше 0). Для завершения введите 0.\n")
    
    numbers = []
    
    try:
        while True:
            try:
                num = float(input("Введите число: "))
                
                if num == 0:
                    break
                elif num < 0:
                    print("Число должно быть больше 0!")
                    continue
                
                numbers.append(num)
                
            except ValueError:
                print("Ошибка: введите корректное число!")
        
        if numbers:
            max_number = max(numbers)
            print(f"\n Введено чисел: {len(numbers)}")
            print(f" Наибольшее число: {max_number}")
            print(f" Все числа: {numbers}")
        else:
            print("\n Вы не ввели ни одного числа!")
            
    except KeyboardInterrupt:
        print("\n Программа прервана")

# find_max_number()