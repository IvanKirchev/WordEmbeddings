import tensorflow as tf
import numpy as np
from itertools import chain
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def preprocess_text(file_path):
    with open(file_path) as file:
        text = file.read().lower()
        text = text.replace('\n', ' ')
        text = text.replace(',', '')
        text = text.replace('(', '')
        text = text.replace(')', '')
        text = text.replace('- ', '')
        text = text.replace('/', '')
        sentences = text.split('. ')
    return sentences

def create_vocab(sentences):
    words = list(chain.from_iterable([sentence.split(' ') for sentence in sentences]))
    vocab = list(sorted(set(filter(lambda x: x != '', words))))
    return vocab

def create_word_mappings(vocab):
    int_to_word = {idx: word for idx, word in enumerate(vocab)}
    word_to_int = {word: idx for idx, word in int_to_word.items()}
    return int_to_word, word_to_int

def generate_training_data(sentences, word_to_int, vocab_size, window_size):
    vocab_one_hot = np.eye(vocab_size)

    data = []
    for sentence in sentences:
        sentence_arr = sentence.split(' ')
        for idx, word in enumerate(sentence_arr):
            for ctx_word in sentence_arr[max(idx - window_size, 0): min(idx + window_size, len(sentence_arr))]:
                if word != ctx_word and word != '' and ctx_word != '':
                    data.append([word, ctx_word])

    x_train = [vocab_one_hot[word_to_int[pair[0]]] for pair in data]
    y_train = [vocab_one_hot[word_to_int[pair[1]]] for pair in data]

    return np.asarray(x_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32)

@tf.function
def train_word2vec_model(X, y, W1, b1, W2, b2, n_iters=10000, optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)):
    for _ in range(n_iters):
        with tf.GradientTape() as tape:
            hidden_representation = tf.add(tf.matmul(X, W1), b1)
            logits = tf.add(tf.matmul(hidden_representation, W2), b2)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            cross_entropy_loss = tf.reduce_mean(loss)

        grads = tape.gradient(cross_entropy_loss, [W1, b1, W2, b2])
        optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2]))

    return W1, b1

def evaluate(W1, b1, word_to_int):
    print('Evaluating: Man - boy ~ Father - son')
    E = (W1 + b1).numpy()

    word_embeddings = E

    man_embd = word_embeddings[word_to_int['man']]
    boy_embd = word_embeddings[word_to_int['boy']]
    father_embd = word_embeddings[word_to_int['father']]
    son_embd = word_embeddings[word_to_int['son']]

    man_boy_norm = np.linalg.norm(man_embd - boy_embd)
    father_son_norm = np.linalg.norm(father_embd - son_embd)

    print("Score: ", man_boy_norm - father_son_norm)

def init_parameters(vocab_size, embedding_dims):
    W1 = tf.Variable(tf.random.normal([vocab_size, embedding_dims]))
    b1 = tf.Variable(tf.random.normal([embedding_dims]))
    W2 = tf.Variable(tf.random.normal([embedding_dims, vocab_size]))
    b2 = tf.Variable(tf.random.normal([vocab_size]))

    return W1, b1, W2, b2

if __name__ == "__main__":
    embedding_dims = 50
    sentences = preprocess_text('corpus.txt')
    vocab = create_vocab(sentences)
    vocab_size = len(vocab)
    int_to_word, word_to_int = create_word_mappings(vocab)
    X, y = generate_training_data(sentences, word_to_int, vocab_size, window_size=5)

    W1, b1, W2, b2 = init_parameters(vocab_size, embedding_dims)

    print("Start Training")
    W1, b1 = train_word2vec_model(X, y, W1, b1, W2, b2, n_iters=100)

    evaluate(W1, b1, word_to_int)