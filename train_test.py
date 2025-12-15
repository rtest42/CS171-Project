import numpy as np
import re


def parse_pems_line_multiple_blocks(line):
    """
    Parse one line containing multiple [ ... ] blocks.
    Returns a 2D array: sensors x time_steps
    """
    # Find all [ ... ] blocks
    blocks = re.findall(r'\[([^\]]+)\]', line)

    matrix = []
    for block in blocks:
        # Remove commas and split by whitespace
        numbers = [float(x) for x in re.split(r'[ ,;]+', block.strip()) if x]
        matrix.append(numbers)

    return np.array(matrix)

def load_dataset(filename: str):
    arr = []
    with open(filename, "r") as f:
        for line in f:
            arr.append(parse_pems_line_multiple_blocks(line))

    return np.array(arr)

def load_labels(filename: str):
    with open(filename, "r") as f:
        line = f.readline().strip()

    line = line.strip('[]')

    # Split by whitespace and convert to integers
    # Convert 1-based MATLAB indexing to 0-based Python
    labels = np.array([int(x) for x in line.split()]) - 1

    return labels
