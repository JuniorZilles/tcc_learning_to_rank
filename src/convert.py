from os import read
from numpy.lib.shape_base import split
import pandas as pd

def convert(path) -> pd.DataFrame:
    lines = read_file(path)
    reg = {'relevance':[], 'qid':[]}#,'docId':[]}
    for l in lines:
        index = l.find('#docid')
        #reg['docId'].append(int(l[index:].split(' ')[-1].replace('\n', '')))
        row = l[:index].split(' ')
        for r in row:
            if ':' in r:
                item = r.split(':')
                if 'f'+item[0] not in reg and item[0] != 'qid':
                    reg['f'+item[0]] = [float(item[1])]
                else:
                    if item[0] == 'qid':
                        reg[item[0]].append(int(item[1]))
                    else:
                        reg['f'+item[0]].append(float(item[1]))                    
            elif '' != r:
                reg['relevance'].append(int(r))
    return pd.DataFrame(reg)
    

def read_file(path) -> list:
    content = []
    with open(path, 'r') as r:
        content = r.readlines()
    return content