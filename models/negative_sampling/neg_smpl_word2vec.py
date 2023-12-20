from models.base_model import BaseModel
import tensorflow as tf
import tqdm
import re
import string
import numpy as np
import io

def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

def custom_loss(x_logit, y_true):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = tf.keras.layers.Embedding(vocab_size,
                                        embedding_dim,
                                        input_length=1,
                                        name="w2v_embedding")
        
        self.context_embedding = tf.keras.layers.Embedding(vocab_size,
                                        embedding_dim,
                                        input_length=num_ns+1)
        
    def call(self, pair):
        target, context = pair

        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)

        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots
    
class NegSamlpingWord2Vec(BaseModel):
    def __init__(self, cfg_path) -> None:
        super().__init__(cfg_path)
        self.SEED = 42
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.BUFFER_SIZE = 10000

        dataset, vectorized_layer = self.load_data()
        self.dataset = dataset

    def __generate_training_data(self, sentances, window_size, vocab_size, num_ns):
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
                    num_sampled=self.config.num_ns,
                    unique=True,
                    range_max=self.config.vocab_size,
                    seed=self.SEED,
                    name='negative_sampling'
                )

                context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
                label = tf.constant([1] + [0]*self.config.num_ns, dtype=tf.int64)


                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        return targets, contexts, labels

    def load_data(self):
        text_ds = tf.data.TextLineDataset([self.config.dataset_path]).filter(lambda x: tf.cast(tf.strings.length(x), bool))

        vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=custom_standardization,
            max_tokens=self.config.vocab_size,
            output_mode='int',
            output_sequence_length=self.config.sequence_length
        )

        print(self.config.batch_size)
        vectorize_layer.adapt(text_ds.batch(self.config.batch_size))

        inverse_vocab = vectorize_layer.get_vocabulary()

        text_vector_ds = text_ds.batch(self.config.batch_size).prefetch(self.AUTOTUNE).map(vectorize_layer).unbatch()

        sentances = list(text_vector_ds.as_numpy_iterator())
        
        (targets, contexts, labels) = self.__generate_training_data(sentances, self.config.window_size, self.config.vocab_size, self.config.num_ns)
        
        targets = np.array(targets)
        contexts = np.array(contexts)
        labels = np.array(labels)

        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.config.batch_size, drop_remainder=True)

        dataset = dataset.cache().prefetch(buffer_size=self.AUTOTUNE)

        return dataset, vectorize_layer

    def train(self):
        print("Start of training")
        model = Word2Vec(self.config.vocab_size, self.config.embedding_dim, self.config.num_ns)

        model.compile(
            optimizer = 'adam', 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy']
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

        model.fit(self.dataset, epochs = 20, callbacks = [tensorboard_callback])
        return model

    def eval(self, model, vectorize_layer):
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