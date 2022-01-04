import numpy as np
import pandas as pd
import json
from read_folder import get_ful_path
from convert import read_group, read_score
from sklearn.metrics import label_ranking_average_precision_score, ndcg_score


def eval():
    paths = get_ful_path('predicted')
    for path in paths:
        f_name = path['nome']
        spl_name = f_name.split('.')
        if len(spl_name) == 3:
            algoritmo = spl_name[0]
            dataset = spl_name[1].upper()
            json_name = f'{algoritmo}.{dataset}'
        else:
            model = spl_name[0]
            algoritmo = spl_name[1]
            dataset = spl_name[2].upper()
            json_name = f'{model}.{algoritmo}.{dataset}'
        pathtrain = f'data/{dataset}'
        label = read_group(f"{pathtrain}/{dataset}.test.label")
        group = read_group(f"{pathtrain}/{dataset}.test.group")
        soma = sum(group)
        score = read_score(path['caminho'])
        pos = 0
        avaliacoes = {'average_precision': [], 'mean_average_precision':0, 'ndcg@1': [], 'mean_ndcg@1':0,
                      'ndcg@3': [],'mean_ndcg@3':0, 'ndcg@5': [], 'mean_ndcg@5':0, 'ndcg@10': [], 'mean_ndcg@10':0}
        for i in group:
            nlabel = [1 if x > 0 else 0 for x in label[pos:pos+i]]
            nscore = [score[pos:pos+i]]
            if len(nlabel) > 1:

                averP = label_ranking_average_precision_score(
                    np.array([nlabel]),  np.array(nscore))
                scores = np.asarray([score[pos:pos+i]])
                true_labels = np.asarray([label[pos:pos+i]])
                ndcg_1 = ndcg_score(true_labels, scores, k=1)
                ndcg_3 = ndcg_score(true_labels, scores, k=3)
                ndcg_5 = ndcg_score(true_labels, scores, k=5)
                ndcg_10 = ndcg_score(true_labels, scores, k=10)
                avaliacoes['average_precision'].append(averP)
                avaliacoes['ndcg@1'].append(ndcg_1)
                avaliacoes['ndcg@3'].append(ndcg_3)
                avaliacoes['ndcg@5'].append(ndcg_5)
                avaliacoes['ndcg@10'].append(ndcg_10)
            pos += i
        avaliacoes['mean_average_precision'] = np.mean(avaliacoes['average_precision'])
        avaliacoes['mean_ndcg@1'] = np.mean(avaliacoes['ndcg@1'])
        avaliacoes['mean_ndcg@3'] = np.mean(avaliacoes['ndcg@3'])
        avaliacoes['mean_ndcg@5'] = np.mean(avaliacoes['ndcg@5'])
        avaliacoes['mean_ndcg@10'] = np.mean(avaliacoes['ndcg@10'])
        with open(f'evaluation/{json_name}.json', 'w') as outfile:
            json.dump(avaliacoes, outfile)


eval()
