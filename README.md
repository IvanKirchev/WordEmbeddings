# Word Embedding

Word Embedding implementations based on Word2Vec and GloVe algorithms

## Usage
Single entry point to the two available models. Configuration is read from config.yaml file located in the source dorectory of the models.
```
python3 main.py [bais | negative_sampling]
```

## From scratch implementation using TF only for the autodiff

The source code for this model is located in word2vec.py

## Tensorflow and Keras implementation of Word2Vec with negative sampling