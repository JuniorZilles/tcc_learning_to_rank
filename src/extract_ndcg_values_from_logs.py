from read_folder import get_ful_path
import json

def extract_ndcg_values_from_logs():
    paths = get_ful_path('train_logs')
    items = {}
    for path in paths:
        f_name = path['nome']
        spl_name = f_name.split('.')
        model = spl_name[1]
        algoritmo = spl_name[2]
        dataset = spl_name[3].upper()
        if dataset not in items:
            items[dataset] = {f'{algoritmo}_{model}': {}}
        elif f'{algoritmo}_{model}' not in items[dataset]:
            items[dataset][f'{algoritmo}_{model}'] = {}
        with open(path['caminho'], 'r') as f:
            lines = f.readlines()
            for line in lines:
                spl_line = line.replace('\t', ' ').replace('\n', '').replace(':', ' ').upper().split(' ')
                for i in range(len(spl_line)):
                    if 'NDCG@' in spl_line[i]:
                        position = spl_line[i].find('NDCG@')
                        index = spl_line[i][position:]
                        value = spl_line[i+1] if spl_line[i+1] != '' else spl_line[i+2]
                        if index not in items[dataset][f'{algoritmo}_{model}']:
                            items[dataset][f'{algoritmo}_{model}'][index] = [float(value)]
                        else:
                            items[dataset][f'{algoritmo}_{model}'][index].append(float(value))
    with open('train.recover.json', 'w') as outfile:
        json.dump(items, outfile)
                
extract_ndcg_values_from_logs()