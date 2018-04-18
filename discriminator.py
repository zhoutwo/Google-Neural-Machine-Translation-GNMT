from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import numpy as np
import tensorflow as tf
import data_utils
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout
from keras.models import Model, load_model


def _get_step_num(fn):
    num = fn[(len(fn) - fn[::-1].index('/')):-3]
    return int(num)

def create_model(num_layers=1, num_gpus=0, num_dict_size=40000, latent_dim=1024, checkpoint_folder=None):
    if checkpoint_folder:
        max = 0
        stored_models = glob.glob(checkpoint_folder + '/*.h5')
        if stored_models:
            for f in stored_models:
                step_num = _get_step_num(f)
                if step_num > max:
                    max = step_num
            print("Reading discriminator model from saved model:", checkpoint_folder + str(max) + '.h5')
            return load_model(checkpoint_folder + str(max) + '.h5')

    inputs = Input(shape=(None,), name='discriminator_Input')
    with tf.device('/cpu:0'):
        discriminator = Embedding(num_dict_size, latent_dim, name='discriminator_Embedding')(inputs)

    if not num_gpus:
        with tf.device('/cpu:0'):
            for i in range(num_layers - 1):
                discriminator = LSTM(latent_dim, name='discriminator_LSTM' + str(i), return_sequences=True)(discriminator)
            discriminator, d_h, d_c = LSTM(latent_dim, name='discriminator_LSTM' + str(num_layers - 1), return_state=True)(discriminator)
    else:
        for i in range(num_layers - 1):
            with tf.device('/device:GPU:' + str(i % num_gpus)):
                discriminator = LSTM(latent_dim, name='discriminator_LSTM' + str(i), return_sequences=True)(discriminator)
        with tf.device('/device:GPU:' + str(num_layers % num_gpus)):
            discriminator, d_h, d_c = LSTM(latent_dim, name='discriminator_LSTM' + str(num_layers - 1), return_state=True)(discriminator)
    output = Concatenate(name='discriminator_Concat')([d_h, d_c])
    output = Dense(int(latent_dim), activation='relu', name='discriminator_FC1')(output)
    output = Dropout(0.2)(output)
    output = Dense(int(latent_dim / 2), activation='relu', name='discriminator_FC2')(output)
    output = Dropout(0.2)(output)
    output = Dense(int(latent_dim / 8), activation='relu', name='discriminator_FC3')(output)
    output = Dropout(0.2)(output)
    output = Dense(2, activation='softmax', name='discriminator_Classification')(output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    return model


def get_disc_input(encoder_in, decoder_in):
    encoder_in = list(encoder_in)
    decoder_in = list(decoder_in)
    # Check if encoder_in has EOS_ID
    if data_utils.EOS_ID in encoder_in:
        print("Warning: EOS_ID in encoder input")
        part1 = encoder_in[:encoder_in.index(data_utils.EOS_ID)]
    elif data_utils.PAD_ID in encoder_in:
        part1 = encoder_in[:encoder_in.index(data_utils.PAD_ID)]
    else:
        part1 = encoder_in[:]

    if not data_utils.EOS_ID in decoder_in:
        print("Warning: EOS_ID NOT in decoder input")
        if data_utils.PAD_ID in decoder_in:
            part2 = decoder_in[:decoder_in.index(data_utils.PAD_ID)+1]
            part2[-1] = data_utils.EOS_ID
        else:
            part2 = decoder_in[:] + [data_utils.EOS_ID]
    else:
        part2 = decoder_in[:decoder_in.index(data_utils.EOS_ID)+1] # Include the EOS token

    assert len(part1) + len(part2) <= 180
    result = np.ones(shape=(180,), dtype=np.int32) * -1
    result[:len(part1)] = part1[:]
    if data_utils.GO_ID in decoder_in:
        result[len(part1):len(part1) + len(part2)] = part2[:]
    else:
        print("Warning: GO_ID NOT in decoder input")
        result[len(part1)] = data_utils.GO_ID
        result[len(part1) + 1:len(part1) + 1 + len(part2)] = part2[:]
        result[len(part1) + 1 + len(part2)] = data_utils.EOS_ID
    result_ls = list(result)
    if -1 in result_ls:
        result = result[:result_ls.index(-1)]
    else:
        result = result[:]
    assert 0 not in result
    return result

