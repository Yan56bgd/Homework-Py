def calculate_statistics(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    min_value = sorted_numbers[0]
    max_value = sorted_numbers[-1]
    
    if n % 2 == 1:
        median = sorted_numbers[n // 2]
    else:
        median = (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
    
    return min_value, max_value, median

numbers = [12, 5, 8, 15, 3, 9, 7, 10, 6]
min_val, max_val, median = calculate_statistics(numbers)

print(f"Минимальное: {min_val}")
print(f"Максимальное: {max_val}")
print(f"Медиана: {median}")