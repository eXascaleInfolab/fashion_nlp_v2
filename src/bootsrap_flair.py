from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.data import Corpus

tagger= SequenceTagger.load('../output/final-model.pt')


# define columns
columns = {0: 'text', 1: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = '../data/1st_iteration'
data_folder_2nd_iteration = '../data/2nd_iteration'
retrained='../data/2nd_iteration/trained_no_char.csv'
# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns)

train_2nd_iter = tagger.predict(corpus.train)
for sentence in train_2nd_iter:
    for token in sentence:
        with open(retrained, 'a') as file:
            file.write(f"{token.text} {token.get_tag('ner').value}\n")
    with open(retrained, 'a') as file:
        file.write('\n')

in_memory = True
corpus: Corpus = ColumnCorpus(data_folder_2nd_iteration, columns)
from flair.trainers.trainer import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train(
    f"../output/no_char_2nd_iter",
    max_epochs=100,
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