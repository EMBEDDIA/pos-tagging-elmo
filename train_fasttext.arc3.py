from numpy.random import seed
seed(3)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Lambda, TimeDistributed, Masking, Average
from tensorflow.keras import optimizers
import keras.backend as K
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from allennlp.commands.elmo import ElmoEmbedder
from math import ceil
from extra.apply_vecmap_transform import vecmap
from posutils import load_data, pad_labels, apply_vecmap, normalize
from tensorflow.keras import activations
from tensorflow.keras.layers import LeakyReLU
import pickle


def embed_fasttext(sentences, embeddings, xlingual):
    max_seqlen = max(len(s) for s in sentences) if sentences else 0
    if max_seqlen == 0:
        return []
    embedded = np.full((len(sentences), max_seqlen, 300), fill_value=-999.)
    emb = []
    if embeddings == 'preembedded':
        emb = sentences
    else:
        for s in sentences:
            emb.append([embeddings[w] if w in embeddings else -999.*np.ones(300) for w in s])
    for x,sentence in enumerate(emb):
        seqlen = len(sentence)
        embedded[x, 0:seqlen, :] = sentence
    if xlingual:
        embedded = apply_mapping(embedded)

    return embedded

def load_fasttext(emb_file):
    embeddings = {}
    with open(emb_file, 'r') as embs:
        embs.readline()
        for line in embs:
            line = line.strip().split()
            try:
                embeddings[line[0]] = np.array([float(i) for i in line[1:]])
            except:
                continue
    return embeddings

def embed_elmo(sentences, elmo_embedder, xlingual, normal=False, lang=''):
    emb_batch = 256
    apply_mapping = apply_vecmap

    if elmo_embedder == 'preembedded':
        max_seqlen = max(len(s[0]) for s in sentences) if sentences else 0
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
        emb = elmo_embedder.embed_batch(sentences)
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
    parser.add_argument("--embs", default=None, type=str, help="ft embeddings, skip if pre-embedded")
    parser.add_argument("--train_len", default=0, type=int, required=True, help="number of tokens in train file")
    parser.add_argument("--eval_len", default=0, type=int, required=True, help="number of tokens in evaluation file")
    parser.add_argument('--mat0', help='mapping matrices for layer0 (.npz), do not specify for monolingual setting')
    parser.add_argument('--mat1', help='mapping matrices for layer1 (.npz), do not specify for monolingual')
    parser.add_argument('--mat2', help='mapping matrices for layer2 (.npz), do not specify for monolingual')
    parser.add_argument('--trlang', default='trg', type=str, help='src or trg when mapping train file language')
    parser.add_argument('--evlang', default='trg', type=str, help='src or trg when mapping eval file language')
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--epoch", default=3, type=int, help="number of epochs to train for")
    parser.add_argument("--save", required=True, type=str, help="Filename to save the POS model to")
    args = parser.parse_args()
    

    def generate_batch_data(inputfile, batch_size, fasttext, args):
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
            xlingual = False #[False,]*3
        while True: # it needs to be infinitely iterable            
            x,y = load_data(inputfile)
            if fasttext == 'preembedded':
                with open(inputfile+'.ft-embs.pickle', 'rb') as f:
                    xembedded = pickle.load(f)
                    xembedded = [sent for batch in xembedded for sent in batch]
            print("INPUT SIZES X AND Y", len(x), len(y), len(x[0]), len(y[0]))
            assert len(x) == len(y)
            newxval = []
            yval = []
            for i in range(len(y)):
                if fasttext == 'preembedded':
                    newxval.append(xembedded[i])
                else:
                    newxval.append(x[i])
                yval.append(y[i])
                assert len(newxval) == len(yval)
                if i > 0 and i % batch_size == 0:
                    xval = embed_fasttext(newxval, fasttext, xlingual)
                    ypadded = pad_labels(yval)
                    yield (np.array(xval), np.array(ypadded))
                    newxval = []
                    yval = []
            if len(newxval) > 0:
                xval = embed_fasttext(newxval, fasttext, xlingual)
                ypadded = pad_labels(yval)
                yield (np.array(xval), np.array(ypadded))

    max_len = None
    if args.embs:
        fasttext = load_fasttext(args.embs)
    else:
        fasttext = 'preembedded'
    # NN
    input_layer = Input(shape=(max_len,300), dtype="float32")
    mask = Masking(mask_value=-999., input_shape=(max_len, 300)) (input_layer)
    bilstm1 = Bidirectional(LSTM(units=512, return_sequences=True)) (mask)
    bilstm1 = LeakyReLU() (bilstm1)
    bilstm2 = Bidirectional(LSTM(units=512, return_sequences=True)) (bilstm1)
    bilstm2 = LeakyReLU() (bilstm2)
    bilstm3 = Bidirectional(LSTM(units=256, return_sequences=True)) (bilstm2)
    bilstm3 = LeakyReLU() (bilstm3)
    fulllayer = TimeDistributed(Dense(64, activation="relu")) (bilstm3)
    out = TimeDistributed(Dense(17, activation="softmax")) (fulllayer)
    
    adam = optimizers.Adam(lr=1e-4)
    model = Model(inputs=input_layer, outputs=out)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit_generator(generate_batch_data(args.train_file, args.bs, fasttext, args), steps_per_epoch=ceil(args.train_len/args.bs), epochs=args.epoch, validation_data=generate_batch_data(args.eval_file, args.bs, fasttext, args), validation_steps=ceil(args.eval_len/args.bs) )

    model.save(args.save)

if __name__ == "__main__":
    main()    
