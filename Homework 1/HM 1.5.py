print("ТАБЛИЦА УМНОЖЕНИЯ ОТ 1 ДО 9")
print()
    
    # Вывод таблицы
for i in range(1, 10):
        for j in range(1, 10):
            result = i * j
            print(f"{i} × {j} = {result:2}", end="   ")
        print()
        print("-" * 90)