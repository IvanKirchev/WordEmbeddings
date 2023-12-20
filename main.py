from models.basic.basic_word2vec import Word2Vec
from models.negative_sampling.neg_smpl_word2vec import NegSamlpingWord2Vec

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Model for learning word embeddings based on word2vec concept'
    )

    parser.add_argument('model_name', choices=['basic', 'negative_sampling'], help='Model to be trainedto be used')
    parser.add_argument('--epochs', default=20)

    args = parser.parse_args()
    model_name = args.model_name
    epochs = args.epochs

    if model_name == 'basic':
        model = Word2Vec('models/basic/config.yaml')
        model.train()
    elif model_name == 'negative_sampling':
        model = NegSamlpingWord2Vec('models/negative_sampling/config.yaml')
        model.train()