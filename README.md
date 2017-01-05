## Neural ParsCit

Neural ParsCit is a citation string parser which parses reference strings into its component tags such as Author, Journal, Location, Date, etc. Neural ParsCit uses Long Short Term Memory (LSTM), a deep learning model to parse the reference strings. This deep learning algorithm is chosen as it is designed to perform sequence-to-sequence labeling tasks such as ours. Input to the model are word embeddings which are vector representation of words. We provide word embeddings as well as character embeddings as input to the network.


## Initial setup

To use the tagger, you need Python 2.7, with Numpy, Theano and Gensim installed.


## Parse citation strings

The fastest way to use the parser is to run state-of-the-art pretrained model as follows:

```
./run.py --model_path models/neuralParsCit/ --pre_emb <vectors.bin> --run shell
./run.py --model_path models/neuralParsCit/ --pre_emb <vectors.bin> --run file -i input_file -o output_file
```
The script can run interactively or input can be passed in a file. In the interactive session, the strings are passed one by one. The result is displayed on standard output. If the file option is chosen, the input is given in a file specified by -i option and the output is stored in the directed file. Using the file option, multiple citation strings can be parsed. 

The state-of-the-art trained model is provided in the models folder and is named neuralParsCit. The binary file for word embeddings is provided in the docker image of the current version of neural ParsCit. The hyper parameter ```discarded``` is the number of embeddings not used in our model. Retained words have a frequency of more than 0 in the ACM citation literature from 1994-2014.


## Train a model

To train your own model, you need to use the train.py script and provide the location of the training, development and testing set:

```
./train.py --train train.txt --dev dev.txt --test test.txt
```

The training script will automatically give a name to the model and store it in ./models/
There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc). To see all parameters, simply run:

```
./train.py --help
```

Input files for the training script have to follow the following format: each word of the citation string and its corresponding tag has to be on a separate line. All citation strings must be separated by a blank line.
