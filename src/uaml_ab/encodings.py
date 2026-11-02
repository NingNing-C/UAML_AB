"""Original positional one-hot representation, compatible with archived weights."""
import numpy as np

AA_TO_INT = {
    "M": 1, "R": 2, "H": 3, "K": 4, "D": 5, "E": 6, "S": 7,
    "T": 8, "N": 9, "Q": 10, "C": 11, "U": 12, "G": 13,
    "P": 14, "A": 15, "V": 16, "I": 17, "F": 18, "Y": 19,
    "W": 20, "L": 21, "O": 22, "X": 23, "Z": 23, "B": 23,
    "J": 23, "start": 24, "stop": 25, "-": 26,
}
aa_to_int = AA_TO_INT


def format_batch_seqs(sequences):
    sequences = list(sequences)
    if not sequences:
        raise ValueError("At least one sequence is required.")
    width = max(map(len, sequences)) + 1
    codes = np.zeros((len(sequences), width), dtype=int)
    for row, sequence in enumerate(sequences):
        codes[row, 0] = AA_TO_INT["start"]
        codes[row, 1:len(sequence) + 1] = [AA_TO_INT[residue] for residue in sequence]
    return codes


def seqs_to_onehot_cnn(sequences):
    codes = format_batch_seqs(sequences)
    features = np.zeros((*codes.shape, 24), dtype=int)
    rows, positions = np.where(codes < 24)
    features[rows, positions, codes[rows, positions]] = 1
    return features


def seqs_to_onehot(sequences):
    positional = seqs_to_onehot_cnn(sequences)
    return positional.reshape(len(positional), -1)
