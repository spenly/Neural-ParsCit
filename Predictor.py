from utils import evaluate, create_input
from model import Model
from loader import augment_with_pretrained, load_sentences
import numpy as np
import itertools
import gensim, re
import json

class Predictor:

	def __init__(self):
		model_path = "final_model"
		self.model = Model(model_path=model_path)
		self.f = self.model.build(training=False, **self.model.parameters)	#needs a better name
		self.model.reload()

		self.model.parameters['pre_emb'] = 'vectors.bin'

		pretrained = gensim.models.word2vec.Word2Vec.load_word2vec_format(self.model.parameters['pre_emb'], binary=True)
		new_weights = self.model.components['word_layer'].embeddings.get_value()

		n_words = len(self.model.id_to_word)
		freq = json.load(open('/home/wenqiang/tagger-master/mydockerbuild2/mydockerbuild/freq', 'r'))
		words = [item[0] for item in freq]

		self.model.id_to_word = {}

		for i in xrange((n_words/2), n_words):
			word = words[i]
			if word in pretrained:
				self.model.id_to_word[i-640780] = word
				new_weights[i-640780] = pretrained[word]
		#        c_found += 1
			elif word.lower() in pretrained:
				self.model.id_to_word[i-640780] = word.lower()
				new_weights[i-640780] = pretrained[word.lower()]
		#        c_lower += 1
			elif re.sub('\d', '0', word.lower()) in pretrained:
				self.model.id_to_word[i-640780] = re.sub('\d', '0', word.lower())
				new_weights[i-640780] = pretrained[
					re.sub('\d', '0', word.lower())
				]
		 #       c_zeros += 1

		self.model.id_to_word[0] = '<UNK>'
		self.model.components['word_layer'].embeddings.set_value(new_weights)
		
		del pretrained
		del new_weights

		self.lower = self.model.parameters['lower']
		self.zeros = self.model.parameters['zeros']

		#Create new mapping because model.id_to_word only is an Ordered dict of only training and testing data

		self.word_to_id = {v:i for i,v in self.model.id_to_word.items()}
		self.char_to_id = {v:i for i,v in self.model.id_to_char.items()}


	def prepare_dataset(self, sentences):
		"""
		Prepare the dataset. Return a list of lists of dictionaries containing:
			- word indexes
			- word char indexes
			- tag indexes
		"""
		def f(x):
			#if zeros:
			return re.sub('\d', '0', x)
			#def f(x): return x.lower() if lower else x
		data = []
		for s in sentences:
			str_words = [w[0] for w in s]
			words = [self.word_to_id[f(w) if f(w) in self.word_to_id else '<UNK>']
					 for w in str_words]
			# Skip characters that are not in the training set
			#for num, word in enumerate(words):
			#        if word < 0:
			#               	words[num]=word_to_id['<UNK>']
			chars = [[self.char_to_id[c] for c in w if c in self.char_to_id]
					 for w in str_words]
			caps = [self.cap_feature(w) for w in str_words]
			data.append({
				'str_words': str_words,
				'words': words,
				'chars': chars,
				'caps': caps,
			})
		return data


	def cap_feature(self, s):
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

	#synchronized block: TO DO
	def parseString(self, string):
		#TO DO
		#To be consumed by web-service
		test_file = "test_file"
		file = open(test_file, 'w')
		file.write('\n'.join(string.encode('utf-8').split()))
		file.close()
		test_sentences = load_sentences(test_file, self.lower, self.zeros)
		data = self.prepare_dataset(test_sentences)
		result = ''
		for citation in data:
			input = create_input(citation, self.model.parameters, False)
			y_pred = np.array(self.f[1](*input))[1:-1]
			tags = []
			for i in xrange(len(y_pred)):
				tags.append(self.model.id_to_tag[y_pred[i]])
			for num, word in enumerate(string.encode('utf-8').split()):
				#print word.decode('utf-8')+'\t'+tags[num]
				result += word.decode('utf-8')+'\t'+tags[num]+'\n'
		return result

if __name__ == '__main__':
	p = Predictor()
	
	while True:
		string = raw_input("Enter the citation string: ").decode('utf-8')
		r = p.parseString(string)
		print(str(r))