def print_multiplication_table():
    print("Таблица умножения:")
    for i in range(1, 10):
        for j in range(1, 10):
            print(f"{i} × {j} = {i*j:2}", end="  ")
        print()

def print_division_table():
    print("Таблица деления:")
    for i in range(1, 10):
        for j in range(1, 10):
            result = i / j
            print(f"{i} ÷ {j} = {result:.1f}", end="  ")
        print()

def print_addition_table():
    print("Таблица сложения:")
    for i in range(1, 10):
        for j in range(1, 10):
            print(f"{i} + {j} = {i+j:2}", end="  ")
        print()

def print_subtraction_table():
    print("Таблица вычитания:")
    for i in range(1, 10):
        for j in range(1, 10):
            result = i - j
            print(f"{i} - {j} = {result:2}", end="  ")
        print()

print("Выберите таблицу:")
print("1 - Умножение")
print("2 - Деление") 
print("3 - Сложение")
print("4 - Вычитание")

choice = input("Введите номер (1-4): ")

if choice == "1":
    print_multiplication_table()
elif choice == "2":
    print_division_table()
elif choice == "3":
    print_addition_table()
elif choice == "4":
    print_subtraction_table()
else:
    print("Неверный выбор")