import numpy as np

def average_list_per_epoch(list, epoch_length):
    # Average the list per epoch
    # This will return a list of length epoch_length
    # with the average of the list for each epoch
    average_list = []
    for i in range(0, len(list), epoch_length):
        average_list.append(sum(list[i:i+epoch_length]) / epoch_length)
    return average_list


def place_digit_at_zero(array_like, digit):
    # Set every zeros in the list to the digit.
    # The will edit the list in place

    if isinstance(array_like, list):
        for i in range(len(array_like)):
            if array_like[i] == 0:
                array_like[i] = digit
    elif isinstance(array_like, np.ndarray):
        array_like[array_like == 0] = digit

