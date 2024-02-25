import re
import numpy as np
from collections import namedtuple, defaultdict
from copy import deepcopy
import pickle
import json
import pke

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F


data_path = '../data/'
dataname = 'NELL'
train_name = 'new1'
WORD_VEC_LEN = 300
n_cluster = 10

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    Every dataset is lower cased
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_factors_NELL():
    rela2doc = dict()
    
    with open(data_path + dataname + "/rela_document.txt") as f_doc:
        lines = f_doc.readlines()
        for num in range(181):
            rela = lines[5*num].strip().split('###')[0].strip()
            description = lines[5*num+1].strip().split('##')[1].strip()
            description = clean_str(description)
            e1 = lines[5*num+2].strip().split('###')[1].strip()
            e1 = clean_str(e1)
            e2 = lines[5*num+3].strip().split('###')[1].strip()
            e2 = clean_str(e2)
            rela2doc[rela] = e1 + ' ' + description + ' ' + description + ' ' + description + ' ' + description + ' ' + description + ' ' + e2

    train2doc = dict()
    with open('../data/NELL/datasplit/' + train_name + '_train_tasks.json', 'r') as fr:
        train_task = json.load(fr)
        print(len(list(train_task.keys())))
        train_rela = list(train_task.keys())
        for rela in train_rela:
            train2doc[rela] = rela2doc[rela]

    rela_list = list()
    doc_list = list()
    rela2ids = dict()
    doc2ids = dict()
    for rela, doc in train2doc.items():
        rela_list.append(rela)
        doc_list.append(doc)
    for i, rela in enumerate(rela_list):
        rela2ids[rela] = int(i)
    for i, doc in enumerate(doc_list):
        doc2ids[doc] = int(i)

    corpus_list = list()
    corpus_split = list()
    vocab = defaultdict(float)

    extractor = pke.unsupervised.MultipartiteRank()

    for i, doc in enumerate(doc_list):
        extractor.load_document(input=doc, language='en')
        extractor.candidate_selection()                     # identify keyphrase candidates
        extractor.candidate_weighting()                     # weight keyphrase candidates
        keyphrases = extractor.get_n_best(n = 3, stemming = False)

        for j, (candidate, score) in enumerate(keyphrases):
            corpus_list.append(candidate)
            candidate_ = candidate.split()
            corpus_split.append(candidate_)
            for word in candidate_:
                vocab[word] += 1

    fvec = '../data/glove/glove.6B.300d.txt'
    corpus2id, word_vecs, corpus_vecs = dict(), dict(), dict()

    with open(fvec) as fp:
        for line in fp:
            words = line.split()
            assert len(words) - 1 == WORD_VEC_LEN
            if words[0] in list(vocab.keys()):
                feat = np.zeros(WORD_VEC_LEN)
                for i in range(WORD_VEC_LEN):
                    feat[i] = float(words[i + 1])
                feat = np.array(feat)
                word_vecs[words[0]] = feat

    for corpus, corpus_ in list(zip(corpus_list, corpus_split)):
        vec = np.zeros(WORD_VEC_LEN, dtype='float32')
        cnt = 0
        for i, word in enumerate(corpus_):
            if word in word_vecs.keys():
                vec += word_vecs[word]
                cnt = cnt + 1
        vec = vec / cnt
        corpus_vecs[corpus] = vec

    W = np.zeros(shape=(len(corpus_list), 300), dtype='float32')
    i = 0
    for corpus in corpus_list:
        W[i] = corpus_vecs[corpus]
        corpus2id[corpus] = i
        i += 1
    #print 'risks:', word_vecs['risks']
    #print 'all', word_vecs['all']

    pickle.dump(corpus2id, open(data_path + dataname + '/' + train_name + '_corpus2id_300_' + dataname +'.pkl', 'wb'))
    np.savez(data_path + dataname + '/' + train_name + '_corpus_WordMatrix_300_' + dataname, W)
    print("Dataset %s ---- word2id size: %d, word matrix size: %s" % (dataname, len(corpus2id), str(W.shape)))

def get_factors_Wiki():
    rela2doc = dict()

    with open(data_path + dataname + "/rela_document.txt") as f_doc:
        lines = f_doc.readlines()
        for num in range(575):
            ent = lines[7 * num].strip().split('###')[0].strip()
            name = lines[7 * num + 2].strip().split('###')[1].strip()
            description = lines[7 * num + 1].strip().split('###')[1].strip()
            description = name + ' ' + description
            description = clean_str(description)
            rela2doc[ent] = description

    train2doc = dict()
    with open('../data/Wiki/datasplit/' + train_name + '_train_tasks.json', 'r') as fr:
        train_task = json.load(fr)
        print(len(list(train_task.keys())))
        train_rela = list(train_task.keys())
        for rela in train_rela:
            train2doc[rela] = rela2doc[rela]

    rela_list = list()
    doc_list = list()
    rela2ids = dict()
    doc2ids = dict()

    for rela, doc in train2doc.items():
        rela_list.append(rela)
        doc_list.append(doc)

    for i, rela in enumerate(rela_list):
        rela2ids[rela] = int(i)

    for i, doc in enumerate(doc_list):
        doc2ids[doc] = int(i)

    corpus_list = list()
    corpus_split = list()
    vocab = defaultdict(float)

    extractor = pke.unsupervised.MultipartiteRank()

    for i, doc in enumerate(doc_list):
        extractor.load_document(input=doc, language='en')
        extractor.candidate_selection()                     # identify keyphrase candidates
        extractor.candidate_weighting()                     # weight keyphrase candidates
        keyphrases = extractor.get_n_best(n = 3, stemming = False)

        for j, (candidate, score) in enumerate(keyphrases):
            corpus_list.append(candidate)
            candidate_ = candidate.split()
            corpus_split.append(candidate_)
            for word in candidate_:
                vocab[word] += 1

    fvec = '../data/glove/glove.6B.300d.txt'
    corpus2id, word_vecs, corpus_vecs = dict(), dict(), dict()

    with open(fvec) as fp:
        for line in fp:
            words = line.split()
            assert len(words) - 1 == WORD_VEC_LEN
            if words[0] in list(vocab.keys()):
                feat = np.zeros(WORD_VEC_LEN)
                for i in range(WORD_VEC_LEN):
                    feat[i] = float(words[i + 1])
                feat = np.array(feat)
                word_vecs[words[0]] = feat

    for corpus, corpus_ in list(zip(corpus_list, corpus_split)):
        vec = np.zeros(WORD_VEC_LEN, dtype='float32')
        cnt = 0
        for i, word in enumerate(corpus_):
            if word in word_vecs.keys():
                vec += word_vecs[word]
                cnt = cnt + 1
        vec = vec / cnt
        corpus_vecs[corpus] = vec

    W = np.zeros(shape=(len(corpus_list), 300), dtype='float32')
    i = 0
    for corpus in corpus_list:
        W[i] = corpus_vecs[corpus]
        corpus2id[corpus] = i
        i += 1
    #print 'risks:', word_vecs['risks']
    #print 'all', word_vecs['all']

    pickle.dump(corpus2id, open(data_path + dataname + '/' + train_name + '_corpus2id_300_' + dataname +'.pkl', 'wb'))
    np.savez(data_path + dataname + '/' + train_name + '_corpus_WordMatrix_300_' + dataname, W)
    print("Dataset %s ---- word2id size: %d, word matrix size: %s" % (dataname, len(corpus2id), str(W.shape)))


def shared_factors_clusting_NELL():
    W = np.load(data_path + dataname + '/' + train_name + '_corpus_WordMatrix_300_NELL.npz')
    
    # corpusMatrix = W['arr_0']
    # for i in range(len(corpusMatrix)):
    #     if np.isnan(corpusMatrix[i]).any():
    #         print(i)
    #         # print(corpusMatrix[i])

    corpusMatrix = W['arr_0'][~np.isnan(W['arr_0']).any(axis=1), :]

    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(corpusMatrix)

    # cntt = defaultdict(int)
    # for lb in kmeans.labels_:
    #     cntt[lb] += 1
    # print(cntt)

    # print(metrics.calinski_harabasz_score(corpusMatrix, kmeans.labels_))
    # print(kmeans.cluster_centers_.shape)

    np.save('../data/NELL/' + train_name + '_kmeans_' + str(n_cluster), kmeans.cluster_centers_)

    km = kmeans.cluster_centers_

    relapath = '../data/NELL/embeddings/rela_matrix.npz'
    relaM = np.load(relapath)['relaM']

    cos_sim = cosine_similarity(relaM, km)
    new_cos_sim = F.softmax(torch.tensor(cos_sim) / 0.1, dim=1)

    new_relaM = np.dot(new_cos_sim, km)

    np.savez('../data/NELL/embeddings/' + train_name + '_rela_matrix', relaM = new_relaM)


def shared_factors_clusting_Wiki():
    W = np.load(data_path + dataname + '/' + train_name + '_corpus_WordMatrix_300_Wiki.npz')
    
    # corpusMatrix = W['arr_0']
    # for i in range(len(corpusMatrix)):
    #     if np.isnan(corpusMatrix[i]).any():
    #         print(i)
    #         # print(corpusMatrix[i])

    corpusMatrix = W['arr_0'][~np.isnan(W['arr_0']).any(axis=1), :]

    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(corpusMatrix)

    # cntt = defaultdict(int)
    # for lb in kmeans.labels_:
    #     cntt[lb] += 1
    # print(cntt)

    # print(metrics.calinski_harabasz_score(corpusMatrix, kmeans.labels_))
    # print(kmeans.cluster_centers_.shape)

    np.save('../data/Wiki/' + train_name + '_kmeans_' + str(n_cluster), kmeans.cluster_centers_)

    km = kmeans.cluster_centers_

    relapath = '../data/Wiki/embeddings/rela_matrix.npz'
    relaM = np.load(relapath)['relaM']

    cos_sim = cosine_similarity(relaM, km)
    new_cos_sim = F.softmax(torch.tensor(cos_sim) / 0.1, dim=1)

    new_relaM = np.dot(new_cos_sim, km)

    np.savez('../data/Wiki/embeddings/' + train_name + '_rela_matrix', relaM = new_relaM)

if __name__ == '__main__':
    if dataname == 'NELL':
        get_factors_NELL()
        shared_factors_clusting_NELL()
    else:
        get_factors_Wiki()
        shared_factors_clusting_Wiki()