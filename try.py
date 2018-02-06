from __future__ import print_function
import numpy as np
import pandas as pd
import os
import time
import pprint
import pickle
import _pickle as cPickle

from keras.callbacks import RemoteMonitor
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from mboxConvert import parseEmails,getEmailStats,mboxToBinaryCSV
from kerasPlotter import Plotter
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM,GRU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

picklefile = 'pickled_emails.pickle'
if os.path.isfile(picklefile):
    with open(picklefile, 'r') as load_from:
      emails = pickle.loads(bytes(load_from))
      print(emails)
else:
    print("hello")