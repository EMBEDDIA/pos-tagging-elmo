from numpy.random import seed
seed(3)
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from keras import optimizers, losses
import keras.backend as K
import torch
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from elmoformanylangs import Embedder
from math import ceil
from extra.apply_vecmap_transform import vecmap
from posutils import load_data, embed_elmo, pad_labels, embed_efml
import pickle
tags = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16}
labels = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

def htanh(a):
    return K.maximum(-1.0, K.minimum(1.0, a))

def csd2(x,y):
    return 1.0*losses.cosine_proximity(x,y)+0.5*losses.mean_absolute_error(x,y)


def generate_batch_data(inputfile, batch_size, args):
    if args.weights:
        elmo = Embedder(args.weights, use_cuda=False)
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
            with open(inputfile+'.efml-embs.pickle', 'rb') as f:
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
                xval0, xval1, xval2 = embed_efml(newxval, elmo, xlingual, args, lang=args.evlang, method=args.mapping)
                yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(yval))
                newxval = []
                yval = []
        if len(newxval) > 0:
            xval0, xval1, xval2 = embed_efml(newxval, elmo, xlingual, args, lang=args.evlang, method=args.mapping)
            yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(yval))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file", default=None, type=str, required=True)
    parser.add_argument("--weights", default=None, type=str, required=False, help="efml weights file")
    parser.add_argument("--test_len", default=0, type=int, required=True)
    parser.add_argument('--mat0', help='mapping matrices for layer0 (.npz), optional')
    parser.add_argument('--mat1', help='mapping matrices for layer1 (.npz), optional')
    parser.add_argument('--mat2', help='mapping matrices for layer2 (.npz), optional')
    parser.add_argument('--evlang', default='src', type=str, help='if mapping with vecmap, was test language "src" or "trg" during mapping')
    parser.add_argument('--mapping','--method', help='mapping method, choose among: vecmap, muse, elmogan. No mapping assumed otherwise.')
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--save", required=True, type=str, help="path to trained efml POS model")
    parser.add_argument("--output", type=str, help="where to save predictions")
    parser.add_argument('--direction', type=int, help="if mapping with elmogan, which direction is used, first lang to second (0) or second to first (1)")
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
