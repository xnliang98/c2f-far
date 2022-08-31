import rouge

def compute_metrics(hypothesis, references, aggregator="Avg"):
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
#                            limit_length=True,
#                            length_limit=100, # default 665
#                            length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.0, # used fro rouge-w, if you want to compute rouge-w, you can use 1.2, which lead to the rouge-l will be wrong [this is a bug] 
                           stemming=True)
    scores = evaluator.get_scores(hypothesis, references)
    
    return scores

def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric.upper(), 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def parallel_process(data):
    all_hypothesis, all_references = data
    return compute_metrics(all_hypothesis, all_references)

def compute_rouge_parallel(all_hypothesis, all_references, aggregator="Avg", ncpus=40):
    total_num = len(all_hypothesis)
    block_size = total_num // ncpus
    
    data = [(all_hypothesis[i * block_size: (i+1) * block_size], all_references[i * block_size: (i+1) * block_size]) for i in range(ncpus)]
    
    from multiprocessing import Pool
    with Pool(ncpus) as pool:
        scores_lst = pool.map(parallel_process, data)

    micro_scores = scores_lst[0]
    scores_lst.pop(0)
    assert len(scores_lst) == ncpus - 1
    for scores in scores_lst:
        for key, value in sorted(scores.items(), key=lambda x: x[0]):
            micro_scores[key]['f'] += value['f']
            micro_scores[key]['p'] += value['p']
            micro_scores[key]['r'] += value['r']
    for key, value in sorted(micro_scores.items(), key=lambda x: x[0]):
        print(prepare_results(key, value['p'] / ncpus, value['r'] / ncpus, value['f'] / ncpus))
    return micro_scores

from utils import load_txt, load_jsonline
from tqdm.auto import tqdm
import sys
dataset = "billsum"
data = load_jsonline(f"data/{dataset}/test.json")
refs = [x['summary'].strip().split("\t") for x in data]
refs = ["\n".join(x) for x in refs]
results_path = sys.argv[1]
#  = load_txt(f"/cfs/cfs-pcgwsz/pcgwsz/geoffliang/project_summ/data/clean_data/{dataset}/test.target")
# cands = load_txt(f"tmp_file/billsum/c2f_results/finetuned.cls.dot-product.tokens_200.f_0.b_0.0.l1_0.5.l2_0.5.txt")
# cands = load_txt("tmp_file/billsum/c2f_results/bert-base-uncased.cls.dot-product.tokens_200.f_0.b_0.0.l1_0.5.l2_0.5.txt")
cands = load_txt(results_path)
cands = ["\n".join(x.strip().split("\t")) for x in cands]
res = compute_rouge_parallel(refs, cands, ncpus=8)
print(res)