import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def make_tokenizer():
    """Creates a 69-character Tensorflow tokenizer for preprocessing text into
    character vectors
    Args:
        None
    Returns:
        A Keras Tokenizer with a specified word index"""

    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789–,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

    tk = Tokenizer(num_words=None, char_level=True)

    char_dict = {}

    for i, char in enumerate(alphabet):
        char_dict[char] = i+1

    tk.word_index = char_dict

    return tk


def text_to_input_vectors(
    df,
    tk,
    input_size,
    input_name
):
    """Function for preprocessing text into character arrays to be used as
    training data for a CharCNN model.
    Args:
        df: DataFrame containing data ready to be preprocessed
        tk: Keras Tokenizer with an already-specified word index
        input_size: Desired length of each character array
        input_name: Name of the DataFrame column containing the to-be-processed text
    Returns:
        A 2D np.ndarray of preprocessed character vectors ready for CharCNN input"""
        
    texts = [s.lower() for s in df[input_name].values]
    sequences = tk.texts_to_sequences(texts)

    data = pad_sequences(
        sequences,
        maxlen=input_size,
        padding='post'
    )

    return data
