def multiplication_table():
    """Вывод таблицы умножения от 1 до 9"""
    print("=== ТАБЛИЦА УМНОЖЕНИЯ ===")
    
    # Классическая таблица умножения 10x10
    print("\nКлассическая таблица умножения:")
    print("     ", end="")
    for i in range(1, 10):
        print(f"{i:4}", end="")
    print("\n    " + "─" * 36)
    
    for i in range(1, 10):
        print(f"{i:2} │", end="")
        for j in range(1, 10):
            print(f"{i*j:4}", end="")
        print()
    
    # Отдельные таблицы
    print("\n" + "=" * 50)
    print("Таблицы умножения по числам:")
    
    for i in range(1, 10):
        print(f"\nТаблица умножения на {i}:")
        print("-" * 25)
        for j in range(1, 10):
            print(f"{i} × {j} = {i*j:2}")
# multiplication_table()