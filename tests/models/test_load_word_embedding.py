import pytest
from model import Model
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors

def test_file_not_found():
    with pytest.raises(IOError):
        Model.load_word_embeddings('this_file_does_not_exist.kv')

def test_load_keyedvectors():
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    word_vectors = model.wv
    fname = get_tmpfile('vectors.kv')
    word_vectors.save(fname)
    word_vectors = KeyedVectors.load(fname, mmap='r')

    assert isinstance(Model.load_word_embeddings(word_vectors), KeyedVectors)
