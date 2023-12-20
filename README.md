# Word Embedding

Word Embedding implementations based on Word2Vec concept. 
Users can replace corpus.txt will most text but beware that I've probably missed not so common symbols during text preprocessing. Feel free to file issues.

## Usage
Single entry point for the two available models. Configuration like learning_rate, epochs, etc. is read from a config.yaml file located in the source directory of a model.

```
python3 main.py [basic | negative_sampling]
```

## From scratch implementation using TF only for the autodiff

* This is oversimplified word2vec implementation using the naive approach of processing the softmax activations on every entry in the vocabulary.

* Evaluation is done by calculating the euclidean distances between the following two pairs of word embeddings (man - boy) and (father - son). I haven't explored this score of recent SOTA word embedding but a good score is assumed to be very close to zero > 0.001.

## Tensorflow and Keras implementation of Word2Vec with negative sampling
* Word2Vec with negative sampling