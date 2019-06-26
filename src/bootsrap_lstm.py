from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.data import Corpus
import argparse
from flair.trainers.trainer import ModelTrainer

def load_data(model,first_iteration,retrained):
    tagger= SequenceTagger.load(model)#'../output/final-model.pt'
    # define columns
    columns = {0: 'text', 1: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = first_iteration #'../data/1st_iteration'

    #retrained= '../data/2nd_iteration/trained_no_char.csv'
    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns)

    train_2nd_iter = tagger.predict(corpus.train)
    for sentence in train_2nd_iter:
        for token in sentence:
            with open(retrained, 'a') as file:
                file.write(f"{token.text} {token.get_tag('ner').value}\n")
        with open(retrained, 'a') as file:
            file.write('\n')
    return tagger

def bootsrap(second_iteration,tagger,output_folder):
    in_memory = True
    columns = {0: 'text', 1: 'ner'}
    data_folder_2nd_iteration = second_iteration #'../data/2nd_iteration'
    corpus: Corpus = ColumnCorpus(data_folder_2nd_iteration, columns)
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    trainer.train(
        output_folder, #f"../output/no_char_2nd_iter"
        max_epochs=epochs,
        patience=3,
        train_with_dev=True,
        checkpoint=True,
        anneal_with_restarts=False,
        mini_batch_size=32,
        embeddings_in_memory=in_memory,
        num_workers=8,
    )

#save the result
#gpu special configuration nothing changing

def parse_args():
    parser = argparse.ArgumentParser(
        description="bootstrap")

    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="folder containing model to be loaded")

    parser.add_argument("--first_iteration",
                        type=str,
                        required=True,
                        help="folder containing first iteration input data")

    parser.add_argument("--second_iteration",
                        type=str,
                        required=True,
                        help="folder containing second iteration input data")

    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help="number of epochs")

    parser.add_argument("--retrained",
                        type=str,
                        required=True,
                        help="path to data trained in the second iteration")

    parser.add_argument("--output_folder",
                        type=str,
                        required=True,
                        help="folder containing output results")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    model = args.model
    first_iteration = args.first_iteration
    second_iteration = args.second_iteration
    epochs = args.epochs
    retrained = args.retrained
    output_folder = args.output_folder
    tagger = load_data(model,first_iteration,retrained)
    bootsrap(second_iteration, tagger, output_folder)