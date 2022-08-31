import numpy as np
from numpy.linalg import norm
import math
import yaml
import os
import sys

from tqdm.auto import tqdm
from itertools import combinations
from multiprocessing import Pool

import torch
import torch.nn as nn

from nltk import sent_tokenize, word_tokenize

from utils import load_jsonline, save_jsonline, load_txt, save_txt, make_dirs

def cos_sim(x, y):
    if x is None or y is None:
        return 0
    return np.dot(x, y) / (norm(x) * norm(y))

def get_sim(doc_embedding, window_size=2):
    sim_lst = []
    for index in range(1, len(doc_embedding)):
        st = index - window_size
        en = index + window_size
        if st < 0:
            st = 0
        sim = cos_sim(np.mean(doc_embedding[st: index], axis=0), np.mean(doc_embedding[index: en], axis=0))
        sim_lst.append(sim)
    assert len(sim_lst) == len(doc_embedding) - 1
    return sim_lst

def smooth_sim(sim_lst, window_size=2):
    new_sim_lst = []
    for index in range(len(sim_lst)):
        st = index - window_size
        en = index + window_size + 1
        if st < 0:
            st = 0
        tmp = np.mean(sim_lst[st: en])
        new_sim_lst.append(tmp)
    return new_sim_lst

def get_depth(sim_lst):
    depths = []
    for index in range(len(sim_lst)):
        left = index - 1
        right = index + 1
        if left < 0:
            left = 0
        if right >= len(sim_lst):
            right = len(sim_lst) - 1
        depth = max(0, sim_lst[left] - sim_lst[index]) + max(0, sim_lst[right] - sim_lst[index])
        depths.append(depth)
    return depths

def get_threshold(depths, beta=-0.5):
    return np.mean(depths) + np.std(depths) * (beta)

def get_segments(depths, beta):
    lst = depths > get_threshold(depths, beta)
    segments = []
    tmp = []
    for index, flag in enumerate(lst):
        tmp.append(index)
        if flag == True:
            segments.append(tmp)
            tmp = []
    return segments

def get_partition(data):
    sentence_embeddings, beta = data
    sim_lst = get_sim(sentence_embeddings)
    sim_lst = smooth_sim(sim_lst)
    depths = get_depth(sim_lst)
    partition_of_document = get_segments(depths, 0.5)
    return partition_of_document

# dataset = sys.argv[1]
# beta = sys.argv[2]
dataset = "billsum"
beta = 1
data_path = "tmp_file/"
model_name = "sbert"
# model_name = "bert-base-uncased"
max_length = 80
doc_embeddings = torch.load(os.path.join(data_path, dataset, "sentence_embeddings", f"test.cls.{model_name}.{max_length}.pt"))

data_iterator = [(x, beta) for x in doc_embeddings]
with Pool(32) as pool:
    partition_of_documents = list(tqdm(pool.imap(get_partition, data_iterator), total=len(data_iterator)))
assert len(partition_of_documents) == len(data_iterator)
partitions = [len(x) for x in partition_of_documents]
print("Average partitions of this dataset:", np.mean(np.array(partitions)))
path = os.path.join(data_path, dataset, "partitions")
make_dirs(path)
torch.save(partition_of_documents, os.path.join(path, f"test.cls.{model_name}.{max_length}.{beta}.pt"))