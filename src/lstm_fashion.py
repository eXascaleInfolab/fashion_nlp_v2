from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import BertEmbeddings
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import (
    StackedEmbeddings,
    WordEmbeddings,
    FlairEmbeddings,
    BytePairEmbeddings,
    CharacterEmbeddings,
    OneHotEmbeddings)
import argparse
from flair.models import SequenceTagger

def load_input_dataset(input_folder,embedding_types):
    # define columns
    columns = {0: 'text', 1: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = input_folder #'../data/lstm_input'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns)

    # 2. what tag do we want to predict?
    tag_type = "ner"

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.item2idx)
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    return embeddings,tag_dictionary,corpus

def train_model(output_folder,embeddings,epochs,tag_dictionary,corpus):
    # initialize sequence tagger
    tag_type = "ner"
    in_memory = True
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        rnn_layers=1,
    )


    # initialize trainer
    from flair.trainers.trainer import ModelTrainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    trainer.train(
        output_folder,
        max_epochs=epochs,
        patience=3,
        train_with_dev=True,
        checkpoint=True,
        anneal_with_restarts=False,
        mini_batch_size=32,
        embeddings_in_memory=in_memory,
        num_workers=8,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="lstm")
    parser.add_argument("--data_folder",
                        type=str,
                        required=True,
                        help="input folder containing training and testing sets")

    parser.add_argument("--embedding",
                        type=str,
                        required=True,
                        help="type of embedding")

    parser.add_argument("--epochs",
                        default=150,
                        type=int,
                        help="number of epochs")

    parser.add_argument("--output_folder",
                        type=str,
                        required=True,
                        help="output folder")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    input_folder = args.data_folder
    embedding = args.embedding
    epochs = args.epochs
    output_folder = args.output_folder
    if (embedding=='no_char'):
        embedding_types = [WordEmbeddings("glove"),]
    elif (embedding=='char'):
        embedding_types = [WordEmbeddings("glove"),CharacterEmbeddings(),]
    elif (embedding=='flair'):
        embedding_types = [WordEmbeddings("glove"),FlairEmbeddings('news-forward-fast'),\
                           FlairEmbeddings('news-backward-fast'),]
    else:
        print ('Please choose one fo the following options for the embedding: no_char,char,flair')
    embeddings,tag_dictionary,corpus = load_input_dataset(input_folder, embedding_types)
    train_model(output_folder,embeddings,epochs,tag_dictionary,corpus)
