from __future__ import print_function
import os
import re
import optparse
import json
import numpy as np
import theano
from gensim.models import KeyedVectors
from utils import evaluate, create_input
from model import Model
from loader import augment_with_pretrained, load_sentences, prepare_dataset


optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model_path", default="",
    help="Model location"
)
optparser.add_option(
    "-e", "--pre_emb", default="",
    help="Pretrained embeddings location"
)
optparser.add_option(
    "-r", "--run", default="shell",
    help="Run interactively (=shell) or using file (=file)"
)
optparser.add_option(
    "-i", "--input_file", default="0",
    help="location of input file"
)
optparser.add_option(
    "-o", "--output_file", default="0",
    help="location of output file"
)
opts = optparser.parse_args()[0]

model = Model(model_path=opts.model_path)
f = model.build(training=False, **model.parameters)
model.reload()

model.parameters['pre_emb'] = opts.pre_emb
pretrained = KeyedVectors.load(model.parameters['pre_emb'], mmap='r')
n_words = len(model.id_to_word)

#only include pretrained embeddings for 640780 most frequent words
words = [item[0] for item in json.load(open('freq', 'r'))]

#Create new mapping because model.id_to_word only is an Ordered dict of only training and testing data
model.id_to_word = {}

discarded = 640780
new_weights = np.empty((n_words - n_words/2 + 1, 500), dtype=theano.config.floatX)
for i in range((n_words/2), n_words):
    word = words[i]
    lower = word.lower()
    digits = re.sub(r'\d', '0', lower)
    idx = i - discarded
    if word in pretrained:
        model.id_to_word[idx] = word
        new_weights[idx] = pretrained[word]
    elif lower in pretrained:
        model.id_to_word[idx] = lower
        new_weights[idx] = pretrained[lower]
    elif digits in pretrained:
        model.id_to_word[idx] = digits
        new_weights[idx] = pretrained[digits]

model.id_to_word[0] = '<UNK>'
#Reset the values of word layer
model.components['word_layer'].embeddings.set_value(new_weights)
#release memory occupied by word embeddings
del pretrained
del new_weights

lower = model.parameters['lower']
zeros = model.parameters['zeros']

word_to_id = {v:i for i,v in model.id_to_word.items()}
char_to_id = {v:i for i,v in model.id_to_char.items()}

while True:
    if opts.run == 'file':
        assert opts.input_file
        assert opts.output_file
        input_file = opts.input_file
        output_file = opts.output_file
        data = open(input_file, 'r').read()
        strings = data.split('\n')
    else:
        string = raw_input("Enter the citation string: ")
        strings = [string]
    test_file = "test_file"
    if os.path.exists(test_file):
        os.remove(test_file)
    file = open(test_file, 'a')
    for string in strings:
        file.write('\n'.join(string.split())+'\n')
    file.close()
    test_sentences = load_sentences(test_file, lower, zeros)
    data = prepare_dataset(test_sentences, word_to_id, char_to_id, lower, True)
    for citation in data:
        inputs = create_input(citation, model.parameters, False)
        y_pred = np.array(f[1](*inputs))[1:-1]
        tags = []
        for i in range(len(y_pred)):
            tags.append(model.id_to_tag[y_pred[i]])
        output = []
        for num, word in enumerate(citation['str_words']):
            output.append(word+'\t'+tags[num])
        if opts.run == 'file':
            file = open(output_file, 'w')
            file.write('\n'.join(output))
            file.close()
        else:
            print('\n'.join(output))
    if opts.run == 'file':
        break
