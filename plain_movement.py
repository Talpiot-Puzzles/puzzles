def refocus_shifts(shift_tuple, delta_height, h_anchor):
    (delta_x, delta_y, theta) = shift_tuple
    refocused_x = -delta_height * delta_x / h_anchor
    refocused_y = -delta_height * delta_y / h_anchor
    return delta_x + refocused_x, delta_y + refocused_y, theta

def calculate_shifts_for_layers(shifts, h_anchor, h_ground, layers_num, thickness):
    shifts_for_all_heights = []
    for _ in range(layers_num):
        delta_height = h_ground - h_anchor
        shifts_for_height = [refocus_shifts(shift_tuple, delta_height, h_anchor) for shift_tuple in shifts]
        shifts_for_all_heights.append(shifts_for_height)
        h_anchor += thickness
    return shifts_for_all_heights