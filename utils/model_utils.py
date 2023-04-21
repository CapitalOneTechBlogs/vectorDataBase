import numpy as np
import os
import json
from tensorflow.keras.layers import Input, Embedding, Activation, Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, ThresholdedReLU
from tensorflow.keras.models import Model


def make_model(
    train_data,
    train_classes,
    test_data,
    test_classes,
    tk
):
    """Create and trains a CharCNN using Keras Tensorflow.
    Args:
        train_data: 2D np.ndarray of preprocessed text vectors for model training
        train_classes: 2D np.ndarray of one-hot encoded labels for each train_data record
        test_data:  Like train_data but for model testing set
        test_classes:   Like test_data but for model testing set
        tk: Tokenizer used for preprocessing the text vectors
    Returns:
        Trained CharCNN model"""
        
    input_size = 256
    vocab_size = len(tk.word_index)
    embedding_size = 69
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, -1],
                   [256, 3, -1],
                   [256, 3, -1],
                   [256, 3, 3]]

    fully_connected_layers = [1024, 1024]
    num_of_classes = 5
    dropout_p = 0.5
    optimizer = 'adam'
    loss = 'categorical_crossentropy'

    # Embedding weights
    embedding_weights = []  # (70, 69)
    embedding_weights.append(np.zeros(vocab_size))  # (0, 69)

    for char, i in tk.word_index.items():  # from index 1 to 69
        onehot = np.zeros(vocab_size)
        onehot[i - 1] = 1
        embedding_weights.append(onehot)

    embedding_weights = np.array(embedding_weights)
    print('Load')

    # Embedding layer Initialization
    embedding_layer = Embedding(vocab_size + 1,
                                embedding_size,
                                input_length=input_size,
                                weights=[embedding_weights])

    # Model Construction
    # Input
    inputs = Input(shape=(input_size,), name='input',
                   dtype='int64')  # shape=(?, 1014)
    # Embedding
    x = embedding_layer(inputs)
    # Conv
    for filter_num, filter_size, pooling_size in conv_layers:
        x = Conv1D(filter_num, filter_size)(x)
        x = Activation('relu')(x)
        if pooling_size != -1:
            x = MaxPooling1D(pool_size=pooling_size)(
                x)  # Final shape=(None, 34, 256)
    x = Flatten()(x)  # (None, 8704)
    # Fully connected layers
    for dense_size in fully_connected_layers:
        x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
        x = Dropout(dropout_p)(x)
    # Output Layer
    predictions = Dense(num_of_classes, activation='softmax')(x)
    # Build model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=[
                  'accuracy'])  # Adam, categorical_crossentropy
    model.summary()

    # Training
    model.fit(train_data, train_classes,
              validation_data=(test_data, test_classes),
              batch_size=128,
              epochs=10,
              verbose=2)

    return model
