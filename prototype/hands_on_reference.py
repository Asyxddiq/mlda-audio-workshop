"""
Reference answers for hands_on.py (facilitators only).

    python prototype/hands_on.py --solution     # runs hands_on.py with these
"""

import numpy as np


def rnn_step(x, h_prev, W_x, W_h, b, memory_on=True):
    """new memory = tanh(W_x . x + W_h . h_prev + b)"""
    old_memory = W_h @ h_prev if memory_on else 0.0
    return np.tanh(W_x @ x + old_memory + b)


def lstm_cell_update(f, i, g, c_prev):
    """notebook = (what you keep) x (old notebook) + (what you add) x (new note)"""
    return f * c_prev + i * g
