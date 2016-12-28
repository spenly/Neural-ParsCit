from utils import evaluate, create_input
from model import Model
from loader import augment_with_pretrained, load_sentences, prepare_dataset
import numpy as np
import itertools
import gensim, re, optparse, json

optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model_path", default="",
    help="Model location"
)
optparser.add_option(
    "-e", "--pre_emb", default="",
    help="Pretrained embeddings location"
)
opts = optparser.parse_args()[0]

model_path = opts.model_path
model = Model(model_path=model_path)
f = model.build(training=False, **model.parameters)
model.reload()

model.parameters['pre_emb'] = opts.pre_emb
pretrained = gensim.models.word2vec.Word2Vec.load_word2vec_format(model.parameters['pre_emb'], binary=True)
new_weights = model.components['word_layer'].embeddings.get_value()
n_words = len(model.id_to_word)

#only include pretrained embeddings for 640780 most frequent words
freq = json.load(open('/home/wenqiang/tagger-master/mydockerbuild2/mydockerbuild/freq', 'r'))
words = [item[0] for item in freq]

#Create new mapping because model.id_to_word only is an Ordered dict of only training and testing data
model.id_to_word = {}

for i in xrange((n_words/2), n_words):
    word = words[i]
    if word in pretrained:
        model.id_to_word[i-640780] = word
        new_weights[i-640780] = pretrained[word]
#        c_found += 1
    elif word.lower() in pretrained:
        model.id_to_word[i-640780] = word.lower()
        new_weights[i-640780] = pretrained[word.lower()]
#        c_lower += 1
    elif re.sub('\d', '0', word.lower()) in pretrained:
        model.id_to_word[i-640780] = re.sub('\d', '0', word.lower())
        new_weights[i-640780] = pretrained[
            re.sub('\d', '0', word.lower())
        ]
 #       c_zeros += 1

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


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

while True:
    string = raw_input("Enter the citation string: ")
    test_file = "test_file"
    file = open(test_file, 'w')
    file.write('\n'.join(string.split(' ')))
    file.close()
    test_sentences = load_sentences(test_file, lower, zeros)
    data = prepare_dataset(test_sentences, word_to_id, char_to_id, lower)
    for citation in data:
        input = create_input(citation, model.parameters, False)
        y_pred = np.array(f[1](*input))[1:-1]
        tags = []
        for i in xrange(len(y_pred)):
            tags.append(model.id_to_tag[y_pred[i]])
        for num, word in enumerate(string.split()):
            print word+'\t'+tags[num]
