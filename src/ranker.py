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


class Ranker(object):
    def __init__(self,
                data_path,
                dataset,
                output_path,
                data_type,
                extract_num=None, 
                num_tokens=None, 
                filter_num=0, 
                beta=0.0, 
                lambda1=0.5, 
                lambda2=0.5,
                alpha=1,
                delta=1,
                processors=8,
                sim_method="dot-product",
                max_length=80):
        self.data_path = data_path
        self.dataset = dataset
        self.output_path = output_path
        self.data_type = data_type
        
        self.sim_method = sim_method

        self.extract_num = extract_num
        self.num_tokens = num_tokens
        self.filter_num = filter_num
        self.max_length = max_length
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta = delta
        self.alpha = alpha

        self.processors = processors
    
    def extract_summary(self, doc_embeddings=None, method='cls', model_name_or_path='sbert'):
        raw_data = os.path.join(self.data_path, self.dataset, f"{self.data_type}.json")
        extracted_list = self.rank(doc_embeddings, method=method, model_name_or_path=model_name_or_path)

        data_iterator = []
        for document in load_jsonline(raw_data):
            article = document['document'].split("\t")
            abstract = document['summary'].split("\t")
            data_iterator.append([article, abstract])
            
        summaries = []
        references = []
        def word_count(summary, num_tokens):
            cnt = 0
            for s in summary:
                cnt += len(word_tokenize(s))
            if cnt > self.num_tokens:
                return False
            return True
        summary_ids = []
        for item, extracted in tqdm(zip(data_iterator, extracted_list), desc="extract", total=len(data_iterator)):
            article, abstract = item
            if len(word_tokenize(" ".join(abstract))) < self.filter_num:   
                continue
            if self.extract_num is not None:
                summary = [article[index] for index in extracted[:self.extract_num]]
                summary_id = extracted[:self.extract_num]
            elif self.num_tokens is not None:
                summary = []
                summary_id = []
                for index in extracted:
                    if not word_count(summary, self.num_tokens):
#                         summary.append(article[index])
#                         summary_id.append(index)
                        break
                    summary.append(article[index])
                    summary_id.append(index)
            else:
                print("Please make sure one of extract_num and extract_tokens is not None.")
                exit(0)
            summaries.append(summary)
            summary_ids.append(summary_id)
            references.append([abstract])
            
        summaries = ["\t".join(x) for x in summaries]
        
        output_dir = os.path.join(self.output_path, self.dataset, "c2f_results")
        make_dirs(output_dir)
        if self.extract_num:
            file_name = f"{model_name_or_path.split('/')[-1]}.{method}.{self.sim_method}.sentences_{self.extract_num}.f_{self.filter_num}.b_{self.beta}.l1_{self.lambda1}.l2_{self.lambda2}"
        else:
            file_name = f"{model_name_or_path.split('/')[-1]}.{method}.{self.sim_method}.tokens_{self.num_tokens}.f_{self.filter_num}.b_{self.beta}.l1_{self.lambda1}.l2_{self.lambda2}"
        save_txt(summaries, os.path.join(output_dir, file_name) + ".txt")
        torch.save(summary_ids, os.path.join(output_dir, file_name) + ".pt")
        
#         import pickle as pkl
#         with open(pjoin(self.data_path, f"{self.data_type}.summary_ids.pkl"), 'wb', ) as f:
#             pkl.dump(summary_ids, f)
#         with open(pjoin(self.data_path, f"{self.data_type}.summaries.pkl"), 'wb', ) as f:
#             pkl.dump(summaries, f)
#         with open(pjoin(self.data_path, f"{self.data_type}.references.pkl"), 'wb', ) as f:
#             pkl.dump(references, f)
#         rouge_result = test_rouge(summaries, references, self.processors)
#         return rouge_result

    def pairdown(self, scores, pair_indice, length):
        out_matrix = np.ones((length, length))
        for pair in pair_indice:
            out_matrix[pair[0][0]][pair[0][1]] = scores[pair[1]]
            out_matrix[pair[0][1]][pair[0][0]] = scores[pair[1]]
        return out_matrix

    def get_similarity_matrix(self, sentence_embeddings):
        pairs = []
        scores = []
        cnt = 0
#         sentence_embeddings = sentence_embeddings[:200]
        for i in range(len(sentence_embeddings)-1):
            for j in range(i, len(sentence_embeddings)):
                if self.sim_method == "dot-product":
                    # computation method 1: dot product 
                    scores.append(np.dot(sentence_embeddings[i], sentence_embeddings[j]))
                elif self.sim_method == "cosin-sim":
                    scores.append(np.dot(sentence_embeddings[i], sentence_embeddings[j]) / (norm(sentence_embeddings[i])* norm(sentence_embeddings[j])))
                elif self.sim_method == "l2":
                    scores.append(norm(sentence_embeddings[i] - sentence_embeddings[j]))
                else:
                    pass
                pairs.append(([i, j], cnt))
                cnt += 1
                                
        return self.pairdown(scores, pairs, len(sentence_embeddings))

    def compute_scores(self, similarity_matrix, edge_threshold=0):
#         alpha = 1 
        n = len(similarity_matrix)
        forward_scores = [1e-10 for i in range(len(similarity_matrix))]
        backward_scores = [1e-10 for i in range(len(similarity_matrix))]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edge_score = similarity_matrix[i][j]
                t = 0
                if edge_score > edge_threshold:
                    t += 1
                    forward_scores[j] += edge_score 
                    backward_scores[i] += edge_score
                if t != 0:
                    forward_scores[j] /= t
                    backward_scores[i] /= t
        return np.asarray(forward_scores), np.asarray(backward_scores), edges
         
    def ranker(self, sentence_embeddings, indexes, flag=False):
        if len(indexes) <=1:
            return indexes
        similarity_matrix = self.get_similarity_matrix(sentence_embeddings)
        min_score = np.min(similarity_matrix)
        max_score = np.max(similarity_matrix)
        threshold = min_score + self.beta * (max_score - min_score)
        
        new_matrix = similarity_matrix - threshold
        for i in range(len(new_matrix)):
            for j in range(len(new_matrix[i])):
                if new_matrix[i][j] <= 0:
                    similarity_matrix[i][j] = 0
        
        forward_score, backward_score, _ = self.compute_scores(new_matrix)

        #forward_score = np.max(forward_score) - forward_score + 1
#         forward_score = 1 - forward_score 

        doc_vector = np.max(sentence_embeddings, axis=0).reshape(-1, 1)

#         scores = [np.dot(summary, doc_vector) for summary in sentence_embeddings]
        scores = [np.dot(summary, doc_vector) / (np.linalg.norm(summary) * np.linalg.norm(doc_vector)) for summary in sentence_embeddings]
        if flag:
            forward_score = 1 - forward_score 
            paired_scores = []
            for index, sc, fs, bs in zip(indexes, scores, forward_score, backward_score):
#                 paired_scores.append([node, scores[node]]) 
#                 paired_scores.append([index,  (sc**2) * np.power((self.lambda1 *  fs + self.lambda2 * bs), self.delta)])
                paired_scores.append([index,  (self.lambda1 * fs + self.lambda2 * bs)])
        else:
            paired_scores = []
            forward_score = 1 - forward_score 
#             forward_score = np.max(forward_score) - forward_score + 1
            for index, sc, fs, bs in zip(indexes, scores, forward_score, backward_score):
#                 paired_scores.append([index,  (sc**self.alpha) * np.power((self.lambda1 *  fs + self.lambda2 * bs), self.delta)])
#                 paired_scores.append([index,  (self.lambda1 * fs + self.lambda2 * bs)])
                paired_scores.append([index,  sc])
            paired_scores.sort(key = lambda x: x[1], reverse = True)
#         return paired_scores
        return [item[0] for item in paired_scores]
        
        
    def two_stage_ranker(self, data):
        sentence_embeddings, group = data
        group_embeddings = [np.mean(sentence_embeddings[g], axis=0) for g in group]
        ranked_gourps = self.ranker(group_embeddings, list(range(len(group))), True)
        # ranked_gourps = ranked_gourps[:int(0.5*len(group))]
        th = 8
        new_sentence_embeddings = []
        new_indexes = []
        for g in ranked_gourps:
            res = group[g]
            
            if len(group[g]) > th:
                res = self.ranker(sentence_embeddings[group[g]], group[g], False)
                res = res[:th]
            new_sentence_embeddings.extend(sentence_embeddings[res])
            new_indexes.extend(res)
        res_2 = self.ranker(new_sentence_embeddings, new_indexes, True)
        return res_2
    
    def rank(self, doc_embeddings=None, method='cls', model_name_or_path='sbert'):
        c = 1
#         doc_embeddings = torch.load(os.path.join(self.output_path, self.dataset, "sentence_embeddings", f"{self.data_type}.{method}.{model_name_or_path.split('/')[-1]}.{self.max_length}.pt"))
        groups = torch.load(os.path.join(self.output_path, self.dataset, "partitions", f"{self.data_type}.{method}.{model_name_or_path.split('/')[-1]}.{self.max_length}.{c}.pt"))
        data_iterator = [(x, y) for x, y in zip(doc_embeddings, groups)]
#         T1 = time.time()
#         for t in tqdm(data_iterator):
#             self.two_stage_ranker(t)
#         print((time.time() - T1))
#         exit(0)
        with Pool(self.processors) as pool:
            extracted_list = list(tqdm(pool.imap(self.two_stage_ranker, data_iterator), total=len(data_iterator), desc='Ranking:'))
        return extracted_list
    
import time     
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__   


with open("yaml_config/ranker.yaml", "r", encoding="utf-8") as f:
    args = dotdict(yaml.load(f, Loader=yaml.FullLoader))
print("Loading Begin!")
doc_embeddings = torch.load(os.path.join(args.output_path, args.dataset, "sentence_embeddings", f"{args.data_type}.{args.method}.{args.model_path.split('/')[-1]}.{args.max_length}.pt"))
print("Loading Done!")

# # main("dev/FAR.yaml")
# with open("dev/FAR.yaml", "r", encoding="utf-8") as f:
#     args = dotdict(yaml.load(f, Loader=yaml.FullLoader))
    
extractor1 = Ranker(data_path=args.data_path,
                    dataset=args.dataset,
                    output_path=args.output_path,
                    data_type=args.data_type,
                    extract_num=args.extract_num, 
                    num_tokens=args.num_tokens, 
                    filter_num=args.filter_num, 
                    beta=args.beta, 
                    lambda1=args.lambda1, 
                    lambda2=args.lambda2,
                    alpha=args.alpha,
                    delta=args.delta,
                    processors=args.processors,
                    sim_method=args.sim_method,
                    max_length=args.max_length)

extractor1.extract_summary(doc_embeddings=doc_embeddings, model_name_or_path=args.model_path.split('/')[-1])