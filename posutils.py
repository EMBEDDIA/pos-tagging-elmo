import csv
import numpy as np
from extra.apply_vecmap_transform import vecmap
from nltk.tokenize import TweetTokenizer
import re

def encode_cat(category):
    y = np.zeros(17)
    tags = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16}
    y[tags[category]] = 1
    return y

def load_data(input_file):
    with open(input_file, "r", encoding='utf-8') as csvfile:
        sentence = []
        labels = []
        X = []
        Y = []
        for row in csvfile:
            row = row.strip().split('\t')
            if len(row) < 2:
                if len(sentence) > 0 and len(sentence) < 1000:
                    X.append(sentence)
                    Y.append(labels)
                sentence = []
                labels = []
            else:
                try:
                    sentence.append(str(row[0]))
                    labels.append(encode_cat(row[1]))
                except:
                    print(row)
        if len(sentence) > 0:
            X.append(sentence)
            Y.append(labels)
    return X,Y


def embed_fasttext(sentences, embeddings, xlingual):
    max_seqlen = max(len(s) for s in sentences) if sentences else 0
    if max_seqlen == 0:
        return []
    max_seqlen = max(max_seqlen, 256)
    embedded = np.full((len(sentences), max_seqlen, 300), fill_value=-999.)
    emb = []
    for s in sentences:
        emb.append([embeddings[w] if w in embeddings else -999.*np.ones(300) for w in s]) 
    for x,sentence in enumerate(emb):
        seqlen = len(sentence)
        embedded[x, 0:seqlen, :] = sentence
    if xlingual:
        embedded = apply_mapping(embedded)
    return embedded

def pad_labels(labels):
    max_seqlen = max(len(s) for s in labels)
    lab0 = np.full((len(labels), max_seqlen, 17), fill_value=-999.)
    for x,label in enumerate(labels):
        seqlen = np.shape(label)[0]
        lab0[x, 0:seqlen, :] = label
    return lab0

def embed_elmo(sentences, elmo_embedder, xlingual, normal=False, lang='', method='vecmap'):
    emb_batch = 256
    if method=='vecmap':
        apply_mapping = apply_vecmap
    elif method=='muse':
        apply_mapping = apply_muse
    elif method=='elmogan':
        return embed_elmogan(sentences, elmo_embedder, xlingual, args)
    else:
        raise ValueError("Unsupported mapping method, use vecmap or muse.")

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

def embed_efml(sentences, elmo_embedder, xlingual, args, normal=False, lang='', method='vecmap'):
    emb_batch = 256
    if method=='vecmap':
        apply_mapping = apply_vecmap
    elif method=='muse':
        apply_mapping = apply_muse
    elif method=='elmogan':
        return embed_elmogan(sentences, elmo_embedder, xlingual, args)
    else:
        raise ValueError("Unsupported mapping method, use vecmap or muse.")
    if elmo_embedder == 'preembedded':
        max_seqlen = max(len(s[0]) for s in sentences) if sentences else 0
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
        max_seqlen = max(max_seqlen, 256)
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


def embed_elmogan(sentences, elmo_embedder, xlingual, args):
    emb_batch = 256
    if elmo_embedder == 'preembedded':
        max_seqlen = max(len(s[0]) for s in sentences) if sentences else 0
        if max_seqlen == 0:
            return []
        max_seqlen = max(max_seqlen, 256)
        emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb = sentences
        sentences = None
    else:
        max_seqlen = max(len(s) for s in sentences) if sentences else 0
        if max_seqlen == 0:
            return []
        max_seqlen = 256
        emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
        emba = []
        emb = elmo_embedder.embed_batch(sentences)
    for x,sentence in enumerate(emb):
        seqlen = sentence[0].shape[0]
        emb0[x, 0:seqlen, :] = elmogan_mapping(sentence[0], xlingual[0], args)
        emb1[x, 0:seqlen, :] = elmogan_mapping(sentence[1], xlingual[1], args)
        emb2[x, 0:seqlen, :] = elmogan_mapping(sentence[2], xlingual[2], args)

    embedded = [emb0, emb1, emb2]

    return embedded

def embed_elmogan5(sentences, elmo_embedder, xlingual, args):
    emb_batch=256
    max_seqlen = max(len(s) for s in sentences) if sentences else 0
    if max_seqlen == 0:
        return []
    max_seqlen = 256
    emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    emba = []
    emb = elmo_embedder.embed_batch(sentences)
    for x,sentence in enumerate(emb):
        seqlen = sentence[0].shape[0]
        concatsent = np.concatenate(sentence, axis=-1)
        mappedsent = elmogan_mapping(concatsent, xlingual, args)
        emb0[x, 0:seqlen, :] = mappedsent[:,:1024]
        emb1[x, 0:seqlen, :] = mappedsent[:,1024:2048]
        emb2[x, 0:seqlen, :] = mappedsent[:,2048:]
    return [emb0, emb1, emb2]

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
    

def apply_vecmap(sentence, W, lang):
    if W:
        mapped_sentence = vecmap(sentence, W[lang], W['s'])
    else:
        mapped_sentence = sentence
    return mapped_sentence

def apply_muse(sentence, W, lang=None):
    mapped_sentence = np.array([np.matmul(W,v) for v in sentence])
    return mapped_sentence

def elmogan_mapping(sentence, W, args):
    if W:
        if args.direction == 0:
            input = [sentence, sentence]
            mapped_sentence, _ = W.predict(input)
        else:
            input = [sentence, sentence]
            _, mapped_sentence = W.predict(input)
    else:
        mapped_sentence = sentence
    if args.normalize:
        normalize(mapped_sentence)
    return mapped_sentence

def normalize(matrix):
    def unit_normalize(matrix):
        norms = np.sqrt(np.sum(matrix**2, axis=1))
        norms[norms == 0] = 1
        matrix /= norms[:, np.newaxis]
    def center_normalize(matrix):
        avg = np.mean(matrix, axis=0)
        matrix -= avg
    unit_normalize(matrix)
    center_normalize(matrix)
    unit_normalize(matrix)
