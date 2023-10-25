def binstr_to_bytes(s: str) -> bytearray:
    """
        Convert a binary string to bytes.

        Parameters:
            s (str): The binary string.

        Returns:
            bytearray: The bytes.
    """
    padding = ''.join('0' for _ in range(8 - len(s) % 8))
    s += padding
    byte = bytearray()
    while len(s) > 0:
        x = int(s[-8:], 2)
        byte.append(x)
        s = s[:-8]
    return byte

def bytes_to_binstr(bytes: bytearray) -> str:
    """
        Convert bytes to a binary string.

        Parameters:
            bytes (bytearray): The bytes.

        Returns:
            str: The binary string.
    """
    s = ''
    for byte in bytes:
        s = bin(byte)[2:].zfill(8) + s
    return s

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

def array_exp_golomb_decoding(number: str) -> list:
    """
        Decode an array of numbers using exponential-Golomb encoding extension to negative numbers.

        Parameters:
            number (str): The number to decode.

        Returns:
            list: The decoded numbers.
    """
    numbers = []
    counter = 0
    pending = ''
    while len(number) > 0:
        current = number[0]
        number = number[1:]
        pending += current
        if current == '0':
            counter += 1
        else:
            for _ in range(counter):
                pending += number[0]
                number = number[1:]
            numbers.append(exp_golomb_decoding(pending))
            pending = ''
            counter = 0
    return numbers
