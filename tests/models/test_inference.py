import os
import tempfile
import pytest
import requests
import json
import re
import numpy as np
from gensim.models import KeyedVectors

from model import Model
from loader import load_sentences, prepare_dataset
from utils import create_input

CORA_URL = "https://raw.githubusercontent.com/knmnyn/ParsCit/master/crfpp/traindata/cora.train"
WORD = Field(init_token='<bos>', eos_token='<eos>')
CHAR_NESTING = Field(tokenize=list, init_token='<bos>', eos_token='<eos>')
CHAR = NestedField(CHAR_NESTING, init_token='<bos>', eos_token='<eos>')
ENTITY = Field(init_token='<bos>', eos_token='<eos>')

# Skip this test when running in CI as the amount of memory is not sufficient
# to build the model
@pytest.mark.skipif(os.getenv("CI") == 'true', reason="Not running in CI")
def test_inference_performance():
    from sklearn.metrics import f1_score
    from torchtext.datasets import SequenceTaggingDataset
    from torchtext.data import Field, NestedField
    
    data_file = tempfile.NamedTemporaryFile(delete=True)

    # TODO Need to be decoded in Python 3
    data_file.write(requests.get(CORA_URL).content)

    fields = [(('text', 'char'), (WORD, CHAR))] + [(None, None)] * 22 + [('entity', ENTITY)]

    dataset = SequenceTaggingDataset(data_file.name, fields, separator=" ")

    model = Model(model_path='models/neuralParsCit')
    model.parameters['pre_emb'] = os.path.join(os.getcwd(), 'vectors.kv') # TODO Change to vectors_with_unk.kv
    f = model.build(training=False, **model.parameters)

    model.reload()

    pretrained = KeyedVectors.load(model.parameters['pre_emb'], mmap='r') # TODO Remove
    n_words = len(model.id_to_word) # TODO Remove
    words = [item[0] for item in json.load(open('freq', 'r'))] # TODO Remove
    model.id_to_word = {} # TODO Remove

    discarded = 640780 # TODO Remove
    new_weights = np.empty((n_words - n_words/2 + 1, 500), dtype='float32') # TODO Remove

    for i in range((n_words/2), n_words): # TODO Remove
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

    model.id_to_word[0] = '<UNK>' # TODO Remove
    model.components['word_layer'].embeddings.set_value(new_weights) # TODO Remove
    del pretrained # TODO Remove
    del new_weights # TODO Remove

    word_to_id = {v:i for i, v in model.id_to_word.items()}
    char_to_id = {v:i for i, v in model.id_to_char.items()}
    tag_to_id = {tag: i for i, tag in model.id_to_tag.items()}

    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.write("\n\n".join(["\n".join(example.text) for example in dataset.examples]))
    tf.close()

    train_sentences = load_sentences(tf.name,
                                     model.parameters['lower'],
                                     model.parameters['zeros'])

    train_inputs = prepare_dataset(train_sentences,
                                   word_to_id,
                                   char_to_id,
                                   model.parameters['lower'], True)

    preds = []

    for citation in train_inputs:
        inputs = create_input(citation, model.parameters, False)
        y_pred = np.array(f[1](*inputs))[1:-1]

        preds.append([(w, y_pred[i]) for i, w in enumerate(citation['str_words'])])

    assert len(preds) == len(dataset.examples)

    for P, T in zip(preds, dataset.examples):
        for p, t in zip(P, zip(T.text, T.entity)):
            results.append((p[1], tag_to_id[t[1]]))

    pred, true = zip(*results)

    eval_metrics = {
        'micro_f1': f1_score(true, pred, average='micro'),
        'macro_f1': f1_score(true, pred, average='macro')
    }

    data_file.close()

    assert eval_metrics == pytest.approx({'macro_f1': 0.98, 'micro_f1': 0.99}, abs=0.01)
