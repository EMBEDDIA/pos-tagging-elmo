from numpy.random import seed
seed(3)
#from tensorflow import set_random_seed
#set_random_seed(3)
#import tensorflow as tf
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Lambda, TimeDistributed, Masking, Average
#from keras_contrib.layers import CRF
#from tensorflow.keras import optimizers
#import keras.backend as K
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
#from allennlp.commands.elmo import ElmoEmbedder
from elmoformanylangs import Embedder
from math import ceil
from extra.apply_vecmap_transform import vecmap
from posutils import load_data, pad_labels, apply_vecmap, normalize
import pickle
#from tensorflow.keras import activations
#from tensorflow.keras.layers import LeakyReLU

def embed_elmo(sentences, elmo_embedder, xlingual, normal=False, lang=''):
    #emb_batch = 256 #128 for et, 2<=n<8 for sv, en=?, others can use higher probably
    #swedish has problem around sentences 1500-1800 in train (extra high ram usage)
    #embedded = map(elmo_embedder.embed_sentence, sentences)
    apply_mapping = apply_vecmap
    emb = []
    max_seqlen = max(len(s) for s in sentences) if sentences else 0
    if max_seqlen == 0:
        return emb
    #emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    #emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    #emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    #emba = []
    emb.append(elmo_embedder.sents2elmo(sentences, output_layer=-2))
    #for x,sentence in enumerate(emb):
    #    if normal:
    #        for i in range(3):
    #            normalize(sentence[i])
    #    seqlen = sentence[0].shape[0]
    #    emb0[x, 0:seqlen, :] = apply_mapping(sentence[0], xlingual[0], lang)
    #    emb1[x, 0:seqlen, :] = apply_mapping(sentence[1], xlingual[1], lang)
    #    emb2[x, 0:seqlen, :] = apply_mapping(sentence[2], xlingual[2], lang)
    #embedded = [emb0, emb1, emb2]
    return emb

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, required=True)
    #parser.add_argument("--options", default=None, type=str, required=True, help="elmo options file")
    parser.add_argument("--weights", default=None, type=str, required=True, help="elmo weights file")
    parser.add_argument("--bs", default=128, type=int, help="batch size")
    args = parser.parse_args()
    

    def generate_batch_data(inputfile, batch_size, args):
        elmo = Embedder(args.weights, use_cuda=True)
        xlingual = [False,]*3
        x,y = load_data(inputfile)
        xemb = []
        print("INPUT SIZES X AND Y", len(x), len(y))
        assert len(x) == len(y)
        newxval = []
        for i in range(len(y)):
            newxval.append(x[i])
            if i > 0 and i % batch_size == 0:
                xemb += embed_elmo(newxval, elmo, xlingual)
                #xemb += xv
                newxval = []
        if len(newxval) > 0:
            xemb += embed_elmo(newxval, elmo, xlingual)
        return xemb

    embs = generate_batch_data(args.train_file, args.bs, args)
    with open(args.train_file+'.efml-embs.pickle', 'wb') as f:
        pickle.dump(embs, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()    
