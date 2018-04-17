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

    inputs = Input(shape=(None,))
    with tf.device('/cpu:0'):
        discriminator = Embedding(num_dict_size, latent_dim, name='discriminator_embedding')(inputs)

    if not num_gpus:
        with tf.device('/cpu:0'):
            for i in range(num_layers - 1):
                discriminator = LSTM(latent_dim, name='discriminator_LSTM' + str(i), return_sequences=True)(
                    discriminator)
            discriminator, d_h, d_c = LSTM(latent_dim, name='discriminator_LSTM' + str(num_layers - 1),
                                           return_state=True)(discriminator)
    else:
        for i in range(num_layers - 1):
            with tf.device('/device:GPU:' + str(i % num_gpus)):
                discriminator = LSTM(latent_dim, name='discriminator_LSTM' + str(i), return_sequences=True)(
                    discriminator)
        with tf.device('/device:GPU:' + str(num_layers % num_gpus)):
            discriminator, d_h, d_c = LSTM(latent_dim, name='discriminator_LSTM' + str(num_layers - 1),
                                           return_state=True)(discriminator)
    output = Concatenate()([d_h, d_c])
    read_model = Model(inputs, output)
    source_inputs = Input(shape=(None,))
    target_inputs = Input(shape=(None,))
    source_output = read_model(source_inputs)
    target_output = read_model(target_inputs)
    fc_input = Concatenate()([source_output, target_output])
    fc_output = Dense(int(latent_dim), activation='relu')(fc_input)
    fc_output = Dropout(0.2)(fc_output)
    fc_output = Dense(int(latent_dim / 2), activation='relu')(fc_output)
    fc_output = Dropout(0.2)(fc_output)
    fc_output = Dense(int(latent_dim / 8), activation='relu')(fc_output)
    fc_output = Dropout(0.2)(fc_output)
    fc_output = Dense(2, activation='softmax', name='discriminator_Classification')(fc_output)

    model = Model(inputs=[source_inputs, target_inputs], outputs=fc_output)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    return model


def get_discriminator_input(input, generated_output, eos_id, sos_id):
    input_length = len(input)
    input_eos_index = input.where(input == eos_id)[0]
    output_eos_index = generated_output(generated_output == eos_id)[0]
    rest_length = input_length - input_eos_index - 1
    if output_eos_index < rest_length:
        print('Warning: output length too large:', input, generated_output, input_eos_index, output_eos_index,
              rest_length)

    input_copy = input[:]
    input_copy[input_eos_index + 1:] = generated_output[:rest_length]
    input_copy[input_eos_index] = sos_id
    return input_copy


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

    # assert len(part1) + len(part2) <= 180
    if len(part1) + len(part2) > 180:
        print("Warning: total length greater than limit:", len(part1) + len(part2))
        part2 = part2[:180-len(part1)]
    result = np.zeros(shape=(180,))
    result[:len(part1)] = part1[:]
    if data_utils.GO_ID in decoder_in:
        result[len(part1):len(part1) + len(part2)] = part2[:]
    else:
        print("Warning: GO_ID NOT in decoder input")
        result[len(part1)] = data_utils.GO_ID
        result[len(part1) + 1:len(part1) + 1 + len(part2)] = part2[:]
        result[len(part1) + 1 + len(part2)] = data_utils.EOS_ID
    return result

def step(step_input, eos_id, sos_id, tf_predicted_output, keras_dis_model, step_output=None):
    """
    Step the model
    Args:
      step_input:
      eos_id:
      tf_predicted_output:
      keras_dis_model:
      step_output:

    Returns:
      True if the given tf_predicted_output is good, False if it needs regenerated

    """

    train_input_false = get_discriminator_input(input=step_input, generated_output=tf_predicted_output, eos_id=eos_id,
                                                sos_id=sos_id)
    if step_output:
        # We are training
        train_input = get_discriminator_input(input=step_input, generated_output=step_output, eos_id=eos_id,
                                              sos_id=sos_id)
        loss1 = keras_dis_model.train_on_batch(np.array([train_input]), np.array([1]))
        loss2 = keras_dis_model.train_on_batch(np.array([train_input_false]), np.array([0]))
        print('Loss 1:', loss1, 'Loss 2:', loss2)

    return keras_dis_model.predict(np.array([train_input_false])) >= 0.5
