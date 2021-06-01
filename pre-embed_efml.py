from numpy.random import seed
seed(3)
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from elmoformanylangs import Embedder
from math import ceil
from extra.apply_vecmap_transform import vecmap
from posutils import load_data, pad_labels, apply_vecmap, normalize
import pickle

def embed_elmo(sentences, elmo_embedder, xlingual, normal=False, lang=''):
    apply_mapping = apply_vecmap
    emb = []
    max_seqlen = max(len(s) for s in sentences) if sentences else 0
    if max_seqlen == 0:
        return emb
    emb.append(elmo_embedder.sents2elmo(sentences, output_layer=-2))
    return emb

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--weights", default=None, type=str, required=True, help="efml weights file")
    parser.add_argument("--bs", default=128, type=int, help="batch size")
    args = parser.parse_args()
    

    def generate_batch_data(inputfile, batch_size, args):
        elmo = Embedder(args.weights, use_cuda=True)
        xlingual = [False,]*3
        x,y = load_data(inputfile)
        xemb = []
        #print("INPUT SIZES X AND Y", len(x), len(y))
        assert len(x) == len(y)
        newxval = []
        for i in range(len(y)):
            newxval.append(x[i])
            if i > 0 and i % batch_size == 0:
                xemb += embed_elmo(newxval, elmo, xlingual)
                newxval = []
        if len(newxval) > 0:
            xemb += embed_elmo(newxval, elmo, xlingual)
        return xemb

    embs = generate_batch_data(args.train_file, args.bs, args)
    with open(args.train_file+'.efml-embs.pickle', 'wb') as f:
        pickle.dump(embs, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
