"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
import seq2seq_model
import data_utils

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("seed", None, "Random seed to use.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("num_gpus", 0, "Number of GPUs available.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("log_dir", "/tmp", "TensorBoard log directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_string("decode_input", None, "Input file to decode.")
tf.app.flags.DEFINE_string("decode_output", None, "Output file to decode to.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(40, 10), (60, 30), (100, 50), (140, 60), (180, 80), (220, 90), (260, 100)]


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only, allow_gpu=True):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.en_vocab_size,
        FLAGS.fr_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.num_gpus if allow_gpu else 0,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    return model


def create_or_load_model(session, model, initial_save=False):
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if initial_save:
            save_checkpoint(session, model)
    return model

def save_checkpoint(sess, model):
    checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
    model.saver.save(sess, checkpoint_path, global_step=model.global_step)


def _get_outputs(ls):
    # This is a greedy decoder - outputs are just argmaxes of eval_output_logits.
    o = [int(np.argmax(l, axis=1)) for l in ls]
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in o:
        o = o[:o.index(data_utils.EOS_ID)]
    return o

def _pad_decode_in(di, s):
    result = np.ones(shape=(s,)) * data_utils.PAD_ID
    result[:len(di)] = di[:]
    return result

def _reset_tf_graph_random_seed():
    if FLAGS.seed is not None:
        tf.set_random_seed(FLAGS.seed)

def _convert_outputs(outputs, rev_vocab):
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
        print("EOS_ID detected in outputs, index:", outputs.index(data_utils.EOS_ID))
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    return " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs])

def train():
    """Train a en->fr translation model using WMT data."""
    # Importing after random seed is set
    import discriminator
    # Prepare WMT data.
    print("Preparing WMT data in %s" % FLAGS.data_dir)
    en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
        FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)

    # Load vocabularies.
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.fr" % FLAGS.fr_vocab_size)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    g_train = tf.Graph()
    g_eval = tf.Graph()
    train_sess = tf.Session(config=config, graph=g_train)
    eval_sess = tf.Session(config=config, graph=g_eval)

    print("Creating discriminator model")
    with tf.device("/cpu:0"):
        dis_model = discriminator.create_model(max_encoder_seq_length=260,
                                               num_layers=FLAGS.num_layers,
                                               num_gpus=FLAGS.num_gpus,
                                               num_dict_size=FLAGS.en_vocab_size,
                                               latent_dim=FLAGS.size,
                                               checkpoint_folder=FLAGS.train_dir)
    writer = None
    summary = None
    with g_train.as_default():
        _reset_tf_graph_random_seed()

        print("Creating model in the training session")
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        with tf.device("/cpu:0"):
            train_model = create_model(train_sess, False)
            train_model = create_or_load_model(train_sess, train_model, initial_save=True)
        writer = tf.summary.FileWriter(logdir=FLAGS.log_dir, graph=g_train)

    with g_eval.as_default():
        _reset_tf_graph_random_seed()

        print("Creating model in the eval session")
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        with tf.device("/cpu:0"):
            eval_model = create_model(eval_sess, True)
            eval_model = create_or_load_model(eval_sess, eval_model)

    # Read data into buckets and compute their sizes.
    print("Reading development and training data (limit: %d)."
          % FLAGS.max_train_data_size)
    dev_set = read_data(en_dev, fr_dev)
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    summary = tf.Summary()
    while True:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in range(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()

        # Normal training step: orig. input
        print("Trainng generator with normal data")
        with g_train.as_default():
            _reset_tf_graph_random_seed()

            encoder_inputs, \
            decoder_inputs, \
            target_weights, \
            original_encoder_inputs, \
            original_decoder_inputs = train_model.get_batch(train_set, bucket_id)
            _, step_loss, _ = train_model.step(train_sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

            summary.value.add(tag="generator_normal_loss", simple_value=step_loss)
            # writer.add_summary(summary, global_step=train_model.global_step.eval(session=train_sess))

        with g_eval.as_default():
            _reset_tf_graph_random_seed()

            original_encoder_inputs_in_original_order = [(list(reversed(oe)), []) for oe in original_encoder_inputs]
            eval_encoder_inputs, eval_decoder_inputs, eval_target_weights, _, _ = eval_model.get_batch(
                {bucket_id: original_encoder_inputs_in_original_order}, bucket_id
            )
            # Get output logits for the sentence.
            _, _, eval_output_logits = eval_model.step(eval_sess, eval_encoder_inputs, eval_decoder_inputs,
                                             eval_target_weights, bucket_id, True)

            # From num_decoder_tokens x batch_size x vocab_size to batch_size x num_decoder_tokens x 1 x vocab_size
            eval_output_logits_transposed = [
                [(logits[i].reshape((1,) + logits[i].shape)) for logits in eval_output_logits]
                for i in range(eval_model.batch_size)
            ]

            for ls in eval_output_logits_transposed:
                print(_get_outputs(ls))

        print("Training discriminator with truth data")
        # From num_encoder_tokens x batch_size to batch_size x num_encoder_tokens
        encoder_inputs_transposed = [
            [row[i] for row in encoder_inputs]
            for i in range(train_model.batch_size)
        ]
        encoder_inputs_transposed_original_order = [
            list(reversed(r)) for r in encoder_inputs_transposed
        ]
        # From num_decoder_tokens x batch_size to batch_size x num_decoder_tokens
        decoder_inputs_transposed = [
            [row[i] for row in decoder_inputs]
            for i in range(train_model.batch_size)
        ]
        disc_in = np.array(
            [discriminator.get_disc_input(encoder_inputs_transposed_original_order[i], decoder_inputs_transposed[i])
            for i in range(train_model.batch_size)]
        )
        # disc_in = np.zeros(shape=(train_model.batch_size,)+)
        disc_out = np.ones(shape=(train_model.batch_size, 1))
        dis_loss = dis_model.train_on_batch(disc_in, disc_out)
        print("Discriminator loss:", dis_loss)

        with g_train.as_default():
            _reset_tf_graph_random_seed()

            summary.value.add(tag="discriminator_truth_loss", simple_value=dis_loss)
            # writer.add_summary(summary, global_step=train_model.global_step.eval(session=train_sess))

        print("Training discriminator with composed data")
        eval_output_tokens = [_pad_decode_in(_get_outputs(ls), 100) for ls in eval_output_logits_transposed]
        composed_disc_in_enc = []
        composed_decoder_out = []
        for i in range(train_model.batch_size):
            e = encoder_inputs_transposed_original_order[i]
            d = eval_output_tokens[i]
            assert data_utils.GO_ID in decoder_inputs_transposed[0]
            dt = [_pad_decode_in(line[1:], 100) for line in decoder_inputs_transposed] # Get rid of GO_ID
            if not np.array_equal(d, dt):
                composed_disc_in_enc.append(e)
                composed_decoder_out.append(d)
        composed_disc_in = np.array(
            [discriminator.get_disc_input(composed_disc_in_enc[i], composed_decoder_out[i])
             for i in range(len(composed_disc_in_enc))]
        )
        composed_decoder_out = np.array(composed_decoder_out)
        composed_disc_out = np.zeros((len(composed_disc_in_enc),))
        composed_dis_loss = dis_model.train_on_batch(composed_disc_in, composed_disc_out)
        print("Discriminator loss:", composed_dis_loss)

        with g_train.as_default():
            _reset_tf_graph_random_seed()

            summary.value.add(tag="discriminator_composed_loss", simple_value=composed_dis_loss)
            writer.add_summary(summary, global_step=train_model.global_step.eval(session=train_sess))
            writer.flush()
            summary = tf.Summary()

        print("Training generator with composed false data")
        with g_train.as_default():
            _reset_tf_graph_random_seed()

            new_enc_in = composed_disc_in
            new_dec_in = [
                _pad_decode_in(line[:line.index(data_utils.EOS_ID)], 100) if data_utils.EOS_ID in line else line
                for line in composed_decoder_out
            ]

            bucket_id_to_use = len(_buckets) - 1

            new_encoder_inputs, \
            new_decoder_inputs, \
            new_target_weights, \
            new_original_encoder_inputs, \
            new_original_decoder_inputs = train_model.get_batch(
                {bucket_id_to_use: [(new_enc_in[i], new_dec_in[i]) for i in range(train_model.batch_size)]}, bucket_id_to_use
            )
            _, new_step_loss, _ = train_model.step(train_sess, new_encoder_inputs, new_decoder_inputs, new_target_weights, bucket_id_to_use,
                                               False)
            print("New step loss:", new_step_loss)
            global_step = train_model.global_step.eval(session=train_sess)

            summary.value.add(tag="generator_composed_loss", simple_value=new_step_loss)
            writer.add_summary(summary, global_step=train_model.global_step.eval(session=train_sess))
            writer.flush()
            summary = tf.Summary()

        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += (step_loss + dis_loss) / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

            with g_train.as_default():
                _reset_tf_graph_random_seed()

                print("global generator step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (global_step, train_model.learning_rate.eval(session=train_sess),
                                step_time, perplexity))
                summary.value.add(tag="learn_rate", simple_value=train_model.learning_rate.eval(session=train_sess))
                summary.value.add(tag="train_step_time", simple_value=step_time)
                summary.value.add(tag="train_perplex", simple_value=perplexity)
                # writer.add_summary(summary, global_step=train_model.global_step.eval(session=train_sess))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                train_sess.run(train_model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss.
            with g_train.as_default():
                _reset_tf_graph_random_seed()

                save_checkpoint(train_sess, train_model)
                dis_model.save(os.path.join(FLAGS.train_dir, str(train_model.global_step.eval(session=train_sess)) + '.h5'))
            # Update eval_model with new weights
            with tf.device("/cpu:0"):
                eval_model = create_or_load_model(eval_sess, eval_model)
            step_time, loss = 0.0, 0.0
            # Run evals on development set and print their perplexity.
            with g_train.as_default():
                _reset_tf_graph_random_seed()

                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, \
                    decoder_inputs, \
                    target_weights, \
                    original_encoder_inputs, \
                    original_decoder_inputs = train_model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = train_model.step(train_sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

                    summary.value.add(tag="eval_perplex_bucket_"+str(bucket_id), simple_value=eval_ppx)
                    writer.add_summary(summary, global_step=train_model.global_step.eval(session=train_sess))
                    writer.flush()
                    summary = tf.Summary()
            sys.stdout.flush()


def decode():
    # Importing after random seed is set
    import discriminator

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            # Create model and load parameters.
            model = create_model(sess, True)
            model = create_or_load_model(sess, model)
            dis_model = discriminator.create_model(max_encoder_seq_length=260,
                                                   num_layers=FLAGS.num_layers,
                                                   num_gpus=0,
                                                   num_dict_size=FLAGS.en_vocab_size,
                                                   latent_dim=FLAGS.size,
                                                   checkpoint_folder=FLAGS.train_dir)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.en" % FLAGS.en_vocab_size)
        fr_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.fr" % FLAGS.fr_vocab_size)
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

        if FLAGS.decode_input:
            with open(FLAGS.decode_input, "r") as input_file:
                with open(FLAGS.decode_input + "_actual", "w") as actual_input_file:
                    with open(FLAGS.decode_output, "w") as output_file:
                        line = input_file.readline()
                        while line:
                            try:
                                # Get token-ids for the input sentence.
                                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(line), en_vocab)
                                # Which bucket does it belong to?
                                bucket_id = min([b for b in range(len(_buckets))
                                                 if _buckets[b][0] >= len(token_ids)])
                                # Get a 1-element batch to feed the sentence to the model.
                                encoder_inputs, \
                                decoder_inputs, \
                                target_weights, \
                                original_encoder_inputs, \
                                original_decoder_inputs = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)

                                max_retries = 10
                                threshold = 0.5
                                for i in range(max_retries):
                                    # Get output logits for the sentence.
                                    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                                     target_weights, bucket_id, True)
                                    # This is a greedy decoder - outputs are just argmaxes of output_logits.
                                    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                                    # If there is an EOS symbol in outputs, cut them at that point.
                                    if data_utils.EOS_ID in outputs:
                                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]

                                    assert model.batch_size == 1

                                    # From num_encoder_tokens x batch_size to batch_size x num_encoder_tokens
                                    encoder_inputs_transposed = [
                                        [row[i] for row in encoder_inputs]
                                        for i in range(model.batch_size)
                                    ]
                                    encoder_inputs_transposed_original_order = [
                                        list(reversed(r)) for r in encoder_inputs_transposed
                                    ]
                                    # From num_decoder_tokens x batch_size to batch_size x num_decoder_tokens
                                    # decoder_inputs_transposed = [
                                    #     [row[i] for row in decoder_inputs]
                                    #     for i in range(model.batch_size)
                                    # ]
                                    output_token_ids = outputs
                                    disc_in = np.array(
                                        [discriminator.get_disc_input(encoder_inputs_transposed_original_order[0],
                                                                      output_token_ids)]
                                    )
                                    disc_out = dis_model.predict(x=disc_in, batch_size=model.batch_size)
                                    disc_out = disc_out[0]
                                    if disc_out > threshold:
                                        break
                                    else:
                                        bucket_id = len(_buckets) - 1
                                        new_enc_in = disc_in

                                        encoder_inputs, \
                                        decoder_inputs, \
                                        target_weights, \
                                        new_original_encoder_inputs, \
                                        new_original_decoder_inputs = model.get_batch(
                                            {bucket_id: [(e, []) for e in new_enc_in]}, bucket_id
                                        )

                                output_file.write(_convert_outputs(outputs, rev_fr_vocab) + "\n")
                                actual_input_file.write(line)
                            except Exception as e:
                                print("Error occurred while decoding", line, ", and the error was:", str(e))
                            line = input_file.readline()

        else:
            # Decode from standard input.
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            while sentence:
                try:
                    # Get token-ids for the input sentence.
                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
                    # Which bucket does it belong to?
                    bucket_id = min([b for b in range(len(_buckets))
                                     if _buckets[b][0] > len(token_ids)])
                    # Get a 1-element batch to feed the sentence to the model.
                    encoder_inputs, \
                    decoder_inputs, \
                    target_weights, \
                    original_encoder_inputs, \
                    original_decoder_inputs = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
                    max_retries = 10
                    threshold = 0.5
                    for i in range(max_retries):
                        # Get output logits for the sentence.
                        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                         target_weights, bucket_id, True)
                        # This is a greedy decoder - outputs are just argmaxes of output_logits.
                        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                        # If there is an EOS symbol in outputs, cut them at that point.
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[:outputs.index(data_utils.EOS_ID)]

                        assert model.batch_size == 1

                        # From num_encoder_tokens x batch_size to batch_size x num_encoder_tokens
                        encoder_inputs_transposed = [
                            [row[i] for row in encoder_inputs]
                            for i in range(model.batch_size)
                        ]
                        encoder_inputs_transposed_original_order = [
                            list(reversed(r)) for r in encoder_inputs_transposed
                        ]
                        # From num_decoder_tokens x batch_size to batch_size x num_decoder_tokens
                        # decoder_inputs_transposed = [
                        #     [row[i] for row in decoder_inputs]
                        #     for i in range(model.batch_size)
                        # ]
                        output_token_ids = outputs
                        disc_in = np.array(
                            [discriminator.get_disc_input(encoder_inputs_transposed_original_order[0],
                                                          output_token_ids)]
                        )
                        disc_out = dis_model.predict(x=disc_in, batch_size=model.batch_size)
                        disc_out = disc_out[0]
                        if disc_out > threshold:
                            break
                        else:
                            bucket_id = len(_buckets) - 1
                            new_enc_in = disc_in

                            encoder_inputs, \
                            decoder_inputs, \
                            target_weights, \
                            new_original_encoder_inputs, \
                            new_original_decoder_inputs = model.get_batch(
                                {bucket_id: [(e, []) for e in new_enc_in]}, bucket_id
                            )
                    # Print out French sentence corresponding to outputs.
                    print(_convert_outputs(outputs, rev_fr_vocab))
                    print("> ", end="")
                    sys.stdout.flush()
                except Exception as e:
                    print("Error occurred while decoding", sentence, ", and the error is:", str(e))
                sentence = sys.stdin.readline()


def self_test():
    """Test the translation model."""
    with tf.Session() as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2, 0,
                                           5.0, 32, 0.3, 0.99, num_samples=8)
        sess.run(tf.global_variables_initializer())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in range(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, \
            decoder_inputs, \
            target_weights, \
            original_encoder_inputs, \
            original_decoder_inputs = model.get_batch(data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                       bucket_id, False)


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
