def convert_color(color_vector):
    color_type = color_vector[3]
    if color_type == 0:
        Y = 0.299 * color_vector[0] + 0.587 * color_vector[1] + 0.114 * color_vector[2]
        I = 0.5959 * color_vector[0] - 0.2746 * color_vector[1] - 0.3213 * color_vector[2]
        Q = 0.2115 * color_vector[0] - 0.5227 * color_vector[1] + 0.3112 * color_vector[2]
        return [Y, I, Q, 1]
    elif color_type == 1:
        R = 1 * color_vector[0] + 0.956 * color_vector[1] + 0.619 * color_vector[2]
        G = 1 * color_vector[0] - 0.272 * color_vector[1] - 0.647 * color_vector[2]
        B = 1 * color_vector[0] - 1.106 * color_vector[1] + 1.703 * color_vector[2]
        return [R, G, B, 0]
    else:
        print("Ошибка! Неверный тип!")
        return None