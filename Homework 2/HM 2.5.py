def replace_negatives(lst):
    result = lst.copy()
    
    for i in range(len(lst)):
        if lst[i] < 0:
            left_pos = None
            for j in range(i-1, -1, -1):
                if lst[j] > 0:
                    left_pos = lst[j]
                    break
            
            right_pos = None
            for j in range(i+1, len(lst)):
                if lst[j] > 0:
                    right_pos = lst[j]
                    break
            
            if left_pos is not None and right_pos is not None:
                result[i] = (left_pos + right_pos) / 2
            elif left_pos is not None:
                result[i] = left_pos
            elif right_pos is not None:
                result[i] = right_pos
            else:
                result[i] = 0
    
    return result

lst = [5, -2, 3, -1, -4, 7, -3, 2]
new_lst = replace_negatives(lst)

print(f"Исходный список: {lst}")
print(f"Результат: {new_lst}")