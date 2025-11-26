import random

def guess_number():
    print("=== ИГРА 'УГАДАЙ ЧИСЛО' ===")
    print("Загадайте число в уме, а я попробую его угадать!")
    print("Отвечайте на вопросы: 'да', 'нет', 'равно'")
    print("-" * 40)
    
    # Запрос интервала у пользователя
    while True:
        try:
            min_num = int(input("Введите начало интервала: "))
            max_num = int(input("Введите конец интервала: "))
            if min_num >= max_num:
                print("Начало интервала должно быть меньше конца!")
                continue
            break
        except ValueError:
            print("Пожалуйста, вводите целые числа!")
    
    attempts = 0
    print(f"\nЯ буду угадывать число от {min_num} до {max_num}")
    print("Отвечайте: 'да', 'нет' или 'равно'\n")
    
    while True:
        attempts += 1
        
        # Метод дихотомии - берем середину интервала
        guess = (min_num + max_num) // 2
        
        # Если интервал сузился до одного числа
        if min_num == max_num:
            print(f"Ваше число: {min_num}!")
            break
        
        print(f"Попытка {attempts}: Число равно {guess}?")
        answer = input("Ваш ответ: ").lower().strip()
        
        if answer in ['равно', 'да', 'yes', 'y', 'correct']:
            print(f"Ура! Я угадал число {guess} за {attempts} попыток!")
            break
        elif answer in ['меньше', 'нет', 'no', 'n', 'less']:
            print(f"Число меньше {guess}?")
            confirm = input("(да/нет): ").lower().strip()
            if confirm in ['да', 'yes', 'y']:
                max_num = guess - 1
            else:
                print("Тогда число больше?")
                confirm2 = input("(да/нет): ").lower().strip()
                if confirm2 in ['да', 'yes', 'y']:
                    min_num = guess + 1
                else:
                    print("Пожалуйста, будьте внимательнее в ответах!")
                    attempts -= 1  # Не засчитываем эту попытку
        elif answer in ['больше', 'more']:
            min_num = guess + 1
        else:
            print("Не понимаю ответ. Используйте: 'да', 'нет', 'равно'")
            attempts -= 1  # Не засчитываем эту попытку
        
        # Проверка на корректность интервала
        if min_num > max_num:
            print("Вы где-то ошиблись в ответах! Давайте начнем заново.")
            return
        
        print(f"Текущий интервал: от {min_num} до {max_num}")
        print("-" * 30)