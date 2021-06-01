from numpy.random import seed
seed(3)
#from tensorflow import set_random_seed
#set_random_seed(3)
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
#from keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Lambda, TimeDistributed, Masking, Average
#from keras_contrib.layers import CRF
from keras import optimizers, losses
import keras.backend as K
import torch
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from allennlp.commands.elmo import ElmoEmbedder
from math import ceil
from extra.apply_vecmap_transform import vecmap
from posutils import load_data, pad_labels, apply_vecmap, apply_muse, embed_elmogan
import pickle
tags = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16}
labels = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

def htanh(a):
    return K.maximum(-1.0, K.minimum(1.0, a))

def csd2(x,y):
    return 1.0*losses.cosine_proximity(x,y)+0.5*losses.mean_absolute_error(x,y)

def embed_elmo(sentences, elmo_embedder, xlingual, args, normal=False, lang='', method='vecmap'):
    emb_batch = 256 #128 for et, 2<=n<8 for sv, en=?, others can use higher probably
    #swedish has problem around sentences 1500-1800 in train (extra high ram usage)
    #embedded = map(elmo_embedder.embed_sentence, sentences)
    if not method or method=='vecmap':
        apply_mapping = apply_vecmap
    elif method=='muse':
        apply_mapping = apply_muse
    elif method=='elmogan':
        return embed_elmogan(sentences, elmo_embedder, xlingual, args)
    else:
        raise ValueError("Unsupported mapping method, use vecmap or muse.")

    if elmo_embedder == 'preembedded':
        #emb = [sent for batch in sentences for sent in batch]
        #print(len(emb), len(emb[0]), len(emb[0][0]))
        #sentences = None
        max_seqlen = max(len(s[0]) for s in sentences) if sentences else 0
        #print(max_seqlen)
        if max_seqlen == 0:
            return []
        max_seqlen = max(max_seqlen, 256)
        emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb = sentences
        sentences = None
    else: # not pre-embedded
        max_seqlen = max(len(s) for s in sentences) if sentences else 0
        if max_seqlen == 0:
            return []
        emb = elmo_embedder.embed_batch(sentences)
        emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    for x,sentence in enumerate(emb):
        #if normal:
        #    for i in range(3):
        #        normalize(sentence[i])
        seqlen = sentence[0].shape[0]
        emb0[x, 0:seqlen, :] = apply_mapping(sentence[0], xlingual[0], lang)
        emb1[x, 0:seqlen, :] = apply_mapping(sentence[1], xlingual[1], lang)
        emb2[x, 0:seqlen, :] = apply_mapping(sentence[2], xlingual[2], lang)
    embedded = [emb0, emb1, emb2]
    return embedded

def generate_batch_data(inputfile, batch_size, args):
    if args.weights:
        elmo = ElmoEmbedder(args.options, args.weights, -1)
    else:
        elmo = 'preembedded'
    if args.mat0 and args.mapping=='vecmap':
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
    elif args.mat0 and args.mapping=='muse':
        M0 = torch.load(args.mat0)
        M1 = torch.load(args.mat1)
        M2 = torch.load(args.mat2)
        xlingual = [M0, M1, M2]
    elif args.mat0 and args.mapping=='elmogan':
        my_funcs = {'htanh':htanh, 'csd2':csd2, 'cosine_proximity':losses.cosine_proximity}
        with tf.device("cpu:0"):
            W0 = load_model(args.mat0, custom_objects=my_funcs)
            W1 = load_model(args.mat1, custom_objects=my_funcs)
            W2 = load_model(args.mat2, custom_objects=my_funcs)
        xlingual = [W0, W1, W2]
    else:
        xlingual = [False,]*3
    while True: # it needs to be infinitely iterable
        x,y = load_data(inputfile)
        if elmo == 'preembedded':
            with open(inputfile+'.elmo-embs.pickle', 'rb') as f:
                xembedded = pickle.load(f)
                xembedded = [sent for batch in xembedded for sent in batch]
        print("INPUT SIZES X AND Y", len(x), len(y))
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
                xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, args, lang=args.evlang, method=args.mapping)
                #ypadded = pad_labels(yval)
                yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(yval))
                newxval = []
                yval = []
        if len(newxval) > 0:
            xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, args, lang=args.evlang, method=args.mapping)
            #ypadded = pad_labels(yval)
            yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(yval))

    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file", default=None, type=str, required=True)
    parser.add_argument("--options", default=None, type=str, required=False, help="elmo options file")
    parser.add_argument("--weights", default=None, type=str, required=False, help="elmo weights file")
    #parser.add_argument("--train_len", default=0, type=int, required=True)
    parser.add_argument("--test_len", default=0, type=int, required=True)
    parser.add_argument('--mat0', help='mapping matrices for layer0 (.npz), optional')
    parser.add_argument('--mat1', help='mapping matrices for layer1 (.npz), optional')
    parser.add_argument('--mat2', help='mapping matrices for layer2 (.npz), optional')
    #parser.add_argument('--trlang', default='trg', type=str, help='src or trg when mapping train file language')
    parser.add_argument('--evlang', default='src', type=str, help='if mapping with vecmap, was test language "src" or "trg" during mapping')
    parser.add_argument('--mapping','--method', help='mapping method, choose among: vecmap, muse, elmogan. No mapping assumed otherwise.')
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--save", default="elmo_new_ner_model", type=str, help="path to trained elmo NER model")
    parser.add_argument("--output", type=str, help="where to save predictions")
    parser.add_argument('--direction', type=int)
    parser.add_argument('--normalize', action="store_true")
    args = parser.parse_args()
    


    max_len = None
    # NN
    model = load_model(args.save)
    y_predict = model.predict_generator(generate_batch_data(args.test_file, args.bs, args), steps=ceil(args.test_len/args.bs))

    _, y_ev = load_data(args.test_file)

    y_ev_i = []
    y_pr_i = []
    for s in range(len(y_ev)):
        for w in range(len(y_ev[s])):
            y_ev_i.append(np.argmax(y_ev[s][w]))
            y_pr_i.append(np.argmax(y_predict[s][w]))

    if args.output:
        with open(args.output, 'w') as writer:
            writer.write('\n'.join(list(map(lambda x: labels[x], y_pr_i))))
    print('---***---')
    print(confusion_matrix(y_ev_i, y_pr_i))
    print(f1_score(y_ev_i, y_pr_i, average='micro'))
    print(f1_score(y_ev_i, y_pr_i, average='macro'))
    print(f1_score(y_ev_i, y_pr_i, average=None))


if __name__ == "__main__":
    main()    
