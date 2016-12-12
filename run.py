from utils import evaluate, create_input
from model import Model
from loader import augment_with_pretrained, load_sentences
import numpy as np
import itertools
import gensim, re

model_path = "/home/wenqiang/tagger-master/models/lower=False,zeros=False,char_dim=25,char_lstm_dim=25,char_bidirect=True,word_dim=500,word_lstm_dim=100,word_bidirect=True,pre_emb=vectors.bin,all_emb=False,cap_dim=1,crf=True,dropout=0.5,lr_method=sgd-lr_.005"
model = Model(model_path=model_path)
f = model.build(training=False, **model.parameters)
model.reload()

pretrained = gensim.models.word2vec.Word2Vec.load_word2vec_format(model.parameters['pre_emb'], binary=True)
new_weights = model.components['word_layer'].embeddings.get_value()

n_words = len(model.id_to_word)

for i in xrange(n_words):
    word = model.id_to_word[i]
    if word in pretrained:
        new_weights[i] = pretrained[word]
#        c_found += 1
    elif word.lower() in pretrained:
        new_weights[i] = pretrained[word.lower()]
#        c_lower += 1
    elif re.sub('\d', '0', word.lower()) in pretrained:
        new_weights[i] = pretrained[
            re.sub('\d', '0', word.lower())
        ]
 #       c_zeros += 1

model.components['word_layer'].embeddings.set_value(new_weights)


lower = model.parameters['lower']
zeros = model.parameters['zeros']
test_file = "/home/wenqiang/tagger-master/test_file"
test_sentences = load_sentences(test_file, lower, zeros)

#Create new mapping because model.id_to_word only is an Ordered dict of only training and testing data
dico = {v:i for i,v in model.id_to_word.items()}

#dico_words, word_to_id, id_to_word = augment_with_pretrained(dico, model.parameters['pre_emb'], list(itertools.chain.from_iterable([[w[0] for w in s] for s in test_sentences])) if not model.parameters['all_emb'] else None)

word_to_id = {v:i for i,v in model.id_to_word.items()}
char_to_id = {v:i for i,v in model.id_to_char.items()}


def prepare_dataset(sentences, word_to_id, char_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
        })
    return data


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


data = prepare_dataset(test_sentences, word_to_id, char_to_id, lower)

for citation in data:
    input = create_input(citation, model.parameters, False)
    y_pred = np.array(f[1](*input))[1:-1]
    tags = []
    for i in xrange(len(y_pred)):
        tags.append(model.id_to_tag[y_pred[i]])
    print tags
