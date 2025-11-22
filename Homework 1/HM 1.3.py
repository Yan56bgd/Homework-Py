def guess_number_game():
    """Программа угадывает число пользователя"""
    print("=== УГАДАЙ ЧИСЛО ===")
    print("Загадайте число, а программа попробует его угадать!\n")
    
    try:
        # Пользователь вводит интервал
        low = int(input("Введите нижнюю границу интервала: "))
        high = int(input("Введите верхнюю границу интервала: "))
        
        if low >= high:
            print("Ошибка: нижняя граница должна быть меньше верхней!")
            return
        
        attempts = 0
        print(f"\n Загадайте число от {low} до {high}...")
        input("Нажмите Enter когда будете готовы...")
        
        while low <= high:
            attempts += 1
            # Метод дихотомии - берем середину интервала
            guess = (low + high) // 2
            
            print(f"\n Попытка #{attempts}")
            print(f"Число равно {guess}?")
            
            answer = input("Введите 'да', 'больше' или 'меньше': ").lower().strip()
            
            if answer == 'да':
                print(f"Я угадал ваше число {guess} за {attempts} попыток!")
                break
            elif answer == 'меньше':
                high = guess - 1
                print(f"Значит число меньше {guess}")
            elif answer == 'больше':
                low = guess + 1
                print(f"Значит число больше {guess}")
            else:
                print("Пожалуйста, введите 'да', 'больше' или 'меньше'")
                attempts -= 1  # Не считаем некорректную попытку
            
            if low > high:
                print("Вы ошиблись с ответами...")
                break
                
    except ValueError:
        print("Ошибка: введите целые числа!")
    except KeyboardInterrupt:
        print("\n Игра прервана")

# guess_number_game()