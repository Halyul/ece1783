import numpy as np

"""
    Encode a number using exponential-Golomb encoding extension to negative numbers.

    Parameters:
        number (int): The number to encode.

    Returns:
        str: The encoded number.
"""
def exp_golomb_encoding(number: int) -> str:
    if number <= 0:
        number = -2 * number
    else:
        number = 2 * number - 1
    number += 1
    binary = bin(number)[2:]
    padding = '0' * (len(binary) - 1)
    return padding + binary

"""
    Decode a number using exponential-Golomb encoding extension to negative numbers.

    Parameters:
        number (str): The number to decode.

    Returns:
        int: The decoded number.
"""  
def exp_golomb_decoding(number: str) -> int:
    padding = number.index('1')
    binary = number[padding:]
    number = int(binary, 2)
    number -= 1
    if number % 2 == 0:
        return -number // 2
    else:
        return (number + 1) // 2

def array_exp_golomb_encoding(array: list) -> list:
    new_array = []
    for item in array:
        new_array.append(exp_golomb_encoding(item))
    return new_array

def array_exp_golomb_decoding(array: list) -> list:
    new_array = []
    for item in array:
        new_array.append(exp_golomb_decoding(item))
    return new_array

"""
    Reordering helper

    Parameters:
        shape (tuple):
"""
def reordering_helper(shape: tuple, current: tuple, row: int) -> tuple:
    if current[1] == row:
        if current[0] == shape[0] - 1:
            row += 1
        current = (row, current[0] + 1)
        if current[1] >= shape[1] - 1:
            current = (row, shape[1] - 1)
    else:
        current = (current[0] + 1, current[1] - 1)
    return current, row

"""
    Matrix reording encoding.

    Parameters:
        qtc_dump (list): The qtc dump to encode.

    Returns:
        list: The encoded qtc dump.
"""
def reording_encoding(qtc_dump: list) -> list:
    array = []
    current = (0, 0)
    row = 0
    end = (len(qtc_dump), len(qtc_dump[0])) # (rows, columns)
    for _ in range(end[0] * end[1]):
        array.append(qtc_dump[current[0]][current[1]])
        current, row = reordering_helper(end, current, row)
    return array

"""
    Matrix reording decoding.

    Parameters:
        reordered_dump (list): The reordered dump to decode.
        shape (tuple): The shape of the matrix.
    
    Returns:
        list: The decoded qtc dump.
"""
def reording_decoding(reordered_dump: list, shape: tuple) -> list:
    matrix = np.empty(shape, dtype=int)
    current = (0, 0)
    row = 0
    for item in reordered_dump:
        matrix[current[0], current[1]] = item
        current, row = reordering_helper(shape, current, row)
    return matrix

"""
    Run-length encoding.

    Parameters:
        array (list): The array to encode.

    Returns:
        list: The encoded array.
"""
def rle_encoding(array: list) -> list:
    new_array = []
    pending = []
    counter = 0
    for item in array:
        if item == 0:
            if counter < 0:
                new_array.append(counter)
                new_array += pending
                pending = []
                counter = 0
            counter += 1
        else:
            if counter > 0:
                new_array.append(counter)
                counter = 0
            counter -= 1
            pending.append(item)
    if len(pending) > 0:
        new_array.append(counter)
        new_array += pending
    new_array.append(0)
    return new_array

"""
    Run-length decoding.

    Parameters:
        array (list): The array to decode.
        shape (tuple): The shape of the matrix.

    Returns:
        list: The decoded array.
"""
def rle_decoding(array: list, shape: tuple) -> list:
    max_length = shape[0] * shape[1]
    new_array = []
    counter = 0
    for i in range(len(array)):
        item = array[i]
        if counter == 0:
            counter = item
            if counter == 0:
                offset = max_length - len(new_array)
                if offset > 0:
                    counter = offset
            if counter > 0:
                for i in range(counter):
                    new_array.append(0)
                counter = 0
            continue
        if counter < 0:
            new_array.append(item)
            counter += 1
        else:
            new_array.append(0)
            counter -= 1
    
    return new_array
