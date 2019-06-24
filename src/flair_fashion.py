from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import BertEmbeddings
from flair.embeddings import ELMoEmbeddings

# define columns
columns = {0: 'text', 1: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = '../data'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns)

from flair.embeddings import (
    StackedEmbeddings,
    WordEmbeddings,
    FlairEmbeddings,
    BytePairEmbeddings,
    CharacterEmbeddings,
    OneHotEmbeddings)

in_memory = True


# 2. what tag do we want to predict?
tag_type = "ner"

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.item2idx)

# initialize embeddings
# initialize embeddings
embedding_types= [

    WordEmbeddings("glove"),

    #BERT
    #https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md
    #BertEmbeddings('bert-base-uncased')

    #ELMO
    #install allennlp
    #ELMoEmbeddings()

    # comment in this line to use trainable one-hot embeddings
    # OneHotEmbeddings(corpus),

    # comment in this line to use byte pair embeddings
    # BytePairEmbeddings('en'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward-fast'),
    # FlairEmbeddings('news-backward-fast'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

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
    f"../output",
    max_epochs=150,
    patience=3,
    train_with_dev=True,
    checkpoint=True,
    anneal_with_restarts=False,
    mini_batch_size=32,
    embeddings_in_memory=in_memory,
    num_workers=8,
)


