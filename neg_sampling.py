import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from word_2_vec import Word2Vec
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

def generate_training_data(sentances, window_size, vocab_size, num_ns, seed = SEED):
    targets, contexts, labels = [], [], []

    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    for sentance in tqdm.tqdm(sentances):

        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence=sentance,
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=0,
            sampling_table=sampling_table
        )
        
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], tf.int64), 1)

            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name='negative_sampling'
            )

            context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype=tf.int64)


            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

text_ds = tf.data.TextLineDataset(['corpus.txt']).filter(lambda x: tf.cast(tf.strings.length(x), bool))

vocab_size = 4096
sequence_length = 10

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

vectorize_layer.adapt(text_ds.batch(1024))

inverse_vocab = vectorize_layer.get_vocabulary()

text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

sentances = list(text_vector_ds.as_numpy_iterator())

num_ns = 20
targets, contexts, labels = generate_training_data(sentances, 10, vocab_size, num_ns)

targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)

BATCH_SIZE = 1024
BUFFER_SIZE = 10000

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

embedding_dim = 128

model = Word2Vec(vocab_size, embedding_dim, num_ns)

model.compile(
    optimizer = 'adam', 
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

model.fit(dataset, epochs = 20, callbacks = [tensorboard_callback])


wegihts = model.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('vectors.tsv', encoding='utf-8', mode='w')
out_m = io.open('metadata.tsv', encoding='utf-8', mode='w')

for idx, word in enumerate(vocab):
    if idx == 0:
        continue

    vec = wegihts[idx]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")

out_v.close()
out_m.close()

