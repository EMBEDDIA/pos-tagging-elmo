from numpy.random import seed
seed(3)
#from tensorflow import set_random_seed
#set_random_seed(3)
#import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Lambda, TimeDistributed, Masking, Average
#from keras_contrib.layers import CRF
from tensorflow.keras import optimizers
import keras.backend as K
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
#from allennlp.commands.elmo import ElmoEmbedder
from elmoformanylangs import Embedder
from math import ceil
from extra.apply_vecmap_transform import vecmap
from posutils import load_data, pad_labels, apply_vecmap, normalize
from tensorflow.keras import activations
from tensorflow.keras.layers import LeakyReLU
import pickle

def embed_elmo(sentences, elmo_embedder, xlingual, normal=False, lang=''):
    emb_batch = 256 #128 for et, 2<=n<8 for sv, en=?, others can use higher probably
    #swedish has problem around sentences 1500-1800 in train (extra high ram usage)
    #embedded = map(elmo_embedder.embed_sentence, sentences)
    apply_mapping = apply_vecmap

    if elmo_embedder == 'preembedded':
        #emb = [sent for batch in sentences for sent in batch]
        #print(len(emb), len(emb[0]), len(emb[0][0]))
        #sentences = None
        max_seqlen = max(len(s[0]) for s in sentences) if sentences else 0
        #print(max_seqlen)
        if max_seqlen == 0:
            return []
        emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb = sentences
        sentences = None
    else: # not pre-embedded
        max_seqlen = max(len(s) for s in sentences) if sentences else 0
        if max_seqlen == 0:
            return []
        emb = elmo_embedder.sents2elmo(sentences, output_layer=-2)
        emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    for x,sentence in enumerate(emb):
        if normal:
            for i in range(3):
                normalize(sentence[i])
        seqlen = sentence[0].shape[0]
        emb0[x, 0:seqlen, :] = apply_mapping(sentence[0], xlingual[0], lang)
        emb1[x, 0:seqlen, :] = apply_mapping(sentence[1], xlingual[1], lang)
        emb2[x, 0:seqlen, :] = apply_mapping(sentence[2], xlingual[2], lang)
    embedded = [emb0, emb1, emb2]
    return embedded

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--eval_file", default=None, type=str, required=True)
    #parser.add_argument("--options", default=None, type=str, help="elmo options file")
    parser.add_argument("--weights", default=None, type=str, help="elmo weights file")
    parser.add_argument("--train_len", default=0, type=int, required=True, help="number of tokens in train file")
    parser.add_argument("--eval_len", default=0, type=int, required=True, help="number of tokens in evaluation file")
    parser.add_argument('--mat0', help='mapping matrices for layer0 (.npz), do not specify for monolingual setting')
    parser.add_argument('--mat1', help='mapping matrices for layer1 (.npz), do not specify for monolingual')
    parser.add_argument('--mat2', help='mapping matrices for layer2 (.npz), do not specify for monolingual')
    parser.add_argument('--trlang', default='trg', type=str, help='src or trg when mapping train file language')
    parser.add_argument('--evlang', default='trg', type=str, help='src or trg when mapping eval file language')
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--epoch", default=3, type=int, help="number of epochs to train for")
    parser.add_argument("--save", default="elmo_new_ner_model", type=str, help="Filename to save the NER model to")
    args = parser.parse_args()
    

    def generate_batch_data(inputfile, batch_size, args):
        if args.weights:
            elmo = Embedder(args.weights, use_cuda=False)
        else:
            elmo = 'preembedded'
        if args.mat0:
            W0 = {}
            W1 = {}
            W2 = {}
            mapmat = np.load(args.mat0)
            W0['src'] = mapmat['wx2']
            W0['trg'] = mapmat['wz2']
            W0['s'] = mapmat['s']
            mapmat = np.load(args.mat1)
            W1['src'] = mapmat['wx2']
            W1['trg'] = mapmat['wz2']
            W1['s'] = mapmat['s']
            mapmat = np.load(args.mat2)
            W2['src'] = mapmat['wx2']
            W2['trg'] = mapmat['wz2']
            W2['s'] = mapmat['s']
            mapmat = None
            xlingual = [W0, W1, W2]
        else:
            xlingual = [False,]*3
        while True: # it needs to be infinitely iterable            
            x,y = load_data(inputfile)
            if elmo == 'preembedded':
                with open(inputfile+'.efml-embs.pickle', 'rb') as f:
                    xembedded = pickle.load(f)
#                   print(len(xembedded), len(xembedded[0]), len(xembedded[0][0]), len(xembedded[0][0][0]))
#                   print(len(xembedded), len(xembedded[1]), len(xembedded[1][0]), len(xembedded[1][0][0]))
#                   print(len(xembedded), len(xembedded[0]), len(xembedded[0][1]), len(xembedded[0][1][0]))
#                   print(len(xembedded), len(xembedded[2]), len(xembedded[0][2]), len(xembedded[0][2][1]))
                    xembedded = [sent for batch in xembedded for sent in batch]
            print("INPUT SIZES X AND Y", len(x), len(y), len(x[0]), len(y[0]))
            assert len(x) == len(y)
            newxval = []
            yval = []
            for i in range(len(y)):
                if elmo == 'preembedded':
                    newxval.append(xembedded[i])
                else:
                    newxval.append(x[i])
                yval.append(y[i])
                assert len(newxval) == len(yval)
                if i > 0 and i % batch_size == 0:
                    if elmo == 'preembedded':
                        xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, args, lang=args.trlang)
                    else:
                        xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, args, lang=args.trlang)
                    ypadded = pad_labels(yval)
                    yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(ypadded))
                    newxval = []
                    yval = []
            if len(newxval) > 0:
                if elmo == 'preembedded':
                    xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, args, lang=args.trlang)
                else:
                    xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, args, lang=args.trlang)
                ypadded = pad_labels(yval)
                yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(ypadded))

    max_len = None
    
    # NN
    #actfun = lambda x: activations.relu(x, alpha=0.5, max_value=30)
    input_cnn = Input(shape=(max_len,1024), dtype="float32")
    mask_cnn = Masking(mask_value=-999., input_shape=(max_len, 1024)) (input_cnn)
    input_lstm1 = Input(shape=(max_len,1024), dtype="float32")
    mask_lstm1 = Masking(mask_value=-999., input_shape=(max_len, 1024)) (input_lstm1)
    input_lstm2 = Input(shape=(max_len,1024), dtype="float32")
    mask_lstm2 = Masking(mask_value=-999., input_shape=(max_len, 1024)) (input_lstm2)
    input_layer = Input(shape=(max_len,1024), dtype="float32")
    avglayer = Average()([mask_cnn, mask_lstm1, mask_lstm2])
    bilstm1 = Bidirectional(LSTM(units=512, return_sequences=True)) (avglayer)
    bilstm1 = LeakyReLU() (bilstm1)
    bilstm2 = Bidirectional(LSTM(units=512, return_sequences=True)) (bilstm1)
    bilstm2 = LeakyReLU() (bilstm2)
    bilstm3 = Bidirectional(LSTM(units=256, return_sequences=True)) (bilstm2)
    bilstm3 = LeakyReLU() (bilstm3)
    fulllayer = TimeDistributed(Dense(64, activation="relu")) (bilstm3)
    out = TimeDistributed(Dense(17, activation="softmax")) (fulllayer)
    
    adam = optimizers.Adam(lr=1e-4)
    model = Model(inputs=[input_cnn, input_lstm1, input_lstm2], outputs=out)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit_generator(generate_batch_data(args.train_file, args.bs, args), steps_per_epoch=ceil(args.train_len/args.bs), epochs=args.epoch, validation_data=generate_batch_data(args.eval_file, args.bs, args), validation_steps=ceil(args.eval_len/args.bs) )

    model.save(args.save)

if __name__ == "__main__":
    main()    
