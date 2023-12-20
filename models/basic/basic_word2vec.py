from models.base_model import BaseModel
import tensorflow as tf
import numpy as np
from itertools import chain

class Word2Vec(BaseModel):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.sentences = self.load_data()
        self.vocab = self.__create_vocab()
        self.vocab_size = len(self.vocab)
        self.int_to_word, self.word_to_int = self.__create_word_mappings()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.embeddings = ()
    
    def __init_parameters(self, embedding_dim):
        W1 = tf.Variable(tf.random.normal([self.vocab_size, embedding_dim]))
        b1 = tf.Variable(tf.random.normal([embedding_dim]))
        W2 = tf.Variable(tf.random.normal([embedding_dim, self.vocab_size]))
        b2 = tf.Variable(tf.random.normal([self.vocab_size]))

        return W1, b1, W2, b2

    def __create_vocab(self):
        words = list(chain.from_iterable([sentence.split(' ') for sentence in self.sentences]))
        vocab = list(sorted(set(filter(lambda x: x != '', words))))
        return vocab

    def __create_word_mappings(self):
        int_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        word_to_int = {word: idx for idx, word in int_to_word.items()}
        return int_to_word, word_to_int

    def __generate_training_data(self, window_size):
        vocab_one_hot = np.eye(self.vocab_size)

        data = []
        for sentence in self.sentences:
            sentence_arr = sentence.split(' ')
            for idx, word in enumerate(sentence_arr):
                for ctx_word in sentence_arr[max(idx - window_size, 0): min(idx + window_size, len(sentence_arr))]:
                    if word != ctx_word and word != '' and ctx_word != '':
                        data.append([word, ctx_word])

        x_train = [vocab_one_hot[self.word_to_int[pair[0]]] for pair in data]
        y_train = [vocab_one_hot[self.word_to_int[pair[1]]] for pair in data]

        return np.asarray(x_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32)


    def __train_word2vec_model(self, X_train, y_train, W1, b1, W2, b2):
        for _ in range(self.config.epochs):
            with tf.GradientTape() as tape:
                hidden_representation = tf.add(tf.matmul(X_train, W1), b1)
                logits = tf.add(tf.matmul(hidden_representation, W2), b2)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=logits)
                cross_entropy_loss = tf.reduce_mean(loss)

            grads = tape.gradient(cross_entropy_loss, [W1, b1, W2, b2])
            self.optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2]))

        return W1, b1
    
    def load_data(self):
        print(self.config.dataset_path)
        with open(self.config.dataset_path) as file:
            text = file.read().lower()
            text = text.replace('\n', ' ')
            text = text.replace(',', '')
            text = text.replace('(', '')
            text = text.replace(')', '')
            text = text.replace('- ', '')
            text = text.replace('/', '')
            sentences = text.split('. ')
        return sentences

    def train(self):
        X_train, y_train = self.__generate_training_data(window_size=self.config.window_size)

        W1, b1, W2, b2 = self.__init_parameters(self.config.embedding_dim)

        print("Start of Training")

        W1, b1 = self.__train_word2vec_model(X_train, y_train, W1, b1, W2, b2)
        self.embeddings = (W1, b1)
        
        self.eval()

    def eval(self):
        print('Evaluating: Man - boy ~ Father - son')
        (W1, b1) = self.embeddings
        E = (W1 + b1).numpy()

        word_embeddings = E

        man_embd = word_embeddings[self.word_to_int['man']]
        boy_embd = word_embeddings[self.word_to_int['boy']]
        father_embd = word_embeddings[self.word_to_int['father']]
        son_embd = word_embeddings[self.word_to_int['son']]

        man_boy_norm = np.linalg.norm(man_embd - boy_embd)
        father_son_norm = np.linalg.norm(father_embd - son_embd)

        print("Score: ", man_boy_norm - father_son_norm)

    def get_embedding(self):
        return self.embeddings