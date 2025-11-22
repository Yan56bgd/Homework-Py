def print_table():
    """Вывод таблицы символов"""
    print("=== ТАБЛИЦА СИМВОЛОВ ===")
    
    # Данные из таблицы
    table_data = [
        [1, 2, 3],
        [4, 5, 6], 
        [7, 10, 9],
        [12, 10, 11]
    ]
    
    # Вывод заголовка
    print("|    | 1   | 2   | 3   |")
    print("|---|---|---|---|")
    
    # Вывод данных
    for i, row in enumerate(table_data, 1):
        # Форматируем каждое число с выравниванием
        formatted_row = [f"{num:^3}" for num in row]
        print(f"|    | {' | '.join(formatted_row)} |")

# print_table()