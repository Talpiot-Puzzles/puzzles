def refocus_shifts(shift_tuple, delta_height, h_anchor):
    """
        Function to refocus the shifts based on the height delta and anchor height.

        Args:
            shift_tuple: Tuple with delta_x, delta_y, and theta values.
            delta_height: Delta of the height.
            h_anchor: Height anchor.

        Returns:
            A tuple of the new delta x, delta y and the same theta.
        """
    (delta_x, delta_y, theta) = shift_tuple
    refocused_x = -delta_height * delta_x / h_anchor
    refocused_y = -delta_height * delta_y / h_anchor
    return delta_x + refocused_x, delta_y + refocused_y, theta

def calculate_shifts_for_layers(shifts, h_anchor, h_ground, layers_num, thickness):
    """
        Function to calculate shifts for all layers.

        Args:
            shifts: A list of shift tuples.
            h_anchor: The anchor height.
            h_ground: The ground height.
            layers_num: The number of layers.
            thickness: The thickness.

        Returns:
            A list of list of shift tuples for each layer height.
        """
    shifts_for_all_heights = []
    for _ in range(layers_num):
        delta_height = h_ground - h_anchor
        shifts_for_height = [refocus_shifts(shift_tuple, delta_height, h_anchor) for shift_tuple in shifts]
        shifts_for_all_heights.append(shifts_for_height)
        h_anchor += thickness
    return shifts_for_all_heights