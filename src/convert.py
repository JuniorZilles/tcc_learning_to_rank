
import numpy as np
import pandas as pd
from pathlib import Path

def read_group(filename: str):
	input = open(filename, "r")
	querygroup = []
	while True:
		line = input.readline()
		if not line:
			break
		querygroup.append(int(line))
	input.close()
	return querygroup

def convert(input_filename, out_data_filename, group_filename) -> list:
	input = open(input_filename, "r")
	output_feature = open(out_data_filename, "w")
	output_group = open(group_filename, "w")
	cur_cnt = 0
	cur_doc_cnt = 0
	last_qid = -1
	while True:
		line = input.readline()
		if not line:
			break
		line = line.split('#')[0]
		tokens = line.split(' ')
		tokens[-1] = tokens[-1].strip()
		label = tokens[0]
		qid = int(tokens[1].split(':')[1])
		if qid != last_qid:
			if cur_doc_cnt > 0:
				output_group.write(str(cur_doc_cnt) + '\n')
				cur_cnt += 1
			cur_doc_cnt = 0
			last_qid = qid
		cur_doc_cnt += 1
		output_feature.write(label+' ')
		output_feature.write(' '.join(tokens[2:]) + '\n')
	output_group.write(str(cur_doc_cnt) + '\n')

	input.close()
	output_feature.close()
	output_group.close()


def toDataframe(input_filename: str):
	input = open(input_filename, "r")
	cur_cnt = 0
	cur_doc_cnt = 0
	last_qid = -1
	querygroup = []
	labels = []
	features = []
	while True:
		line = input.readline()
		if not line:
			break
		tokens = line.split(' ')
		tokens[-1] = tokens[-1].strip()
		label = tokens[0]
		qid = int(tokens[1].split(':')[1])
		if qid != last_qid:
			if cur_doc_cnt > 0:
				querygroup.append(int(cur_doc_cnt))
				cur_cnt += 1
			cur_doc_cnt = 0
			last_qid = qid
		cur_doc_cnt += 1
		labels.append(int(label))
		inner = {}
		for a in tokens[2:]:
			if a != '':
				vl = a.split(':')
				if len(vl) == 2:
					inner[vl[0]] = float(vl[1])
		features.append(inner)
	querygroup.append(cur_doc_cnt)
	df = pd.DataFrame(features)
	input.close()
	labelsnp = np.array(labels)
	return querygroup, labelsnp, df

def toOrdenedFile(input: str, output:str):
	input = open(input, "r")
	itens = {}
	while True:
		line = input.readline()
		if not line:
			break
		tokens = line.split(' ')
		tokens[-1] = tokens[-1].strip()
		qid = int(tokens[1].split(':')[1])
		if qid not in itens:
			itens[qid] = [line]
		else:
			itens[qid].append(line)
	input.close()
	with open(output, "a") as wr:
		keys = list(itens.keys())
		keys.sort()
		for qid in keys:
			for line in itens[qid]:
				wr.write(line)
	

def transform_lightgbm_flaml():
	for data in ['MSLR10K', 'MSLR30K', 'OHSUMED', 'TD2003', 'TD2004']:
		pathtrain = Path(__file__).absolute().parents[1] / 'data' / data
		train = str(pathtrain/f"{data}.train")
		test = str(pathtrain/f"{data}.test")
		vali = str(pathtrain/f"{data}.vali")
		train_group = str(pathtrain/f"{data}.train.group")
		test_group = str(pathtrain/f"{data}.test.group")
		vali_group = str(pathtrain/f"{data}.vali.group")
		convert(str(pathtrain/'train.txt'), train, train_group)
		convert(str(pathtrain/'test.txt'), test, test_group)
		convert(str(pathtrain/'vali.txt'), vali, vali_group)

def transform_ranksvm():
	for data in ['MSLR10K', 'MSLR30K', 'OHSUMED', 'TD2003', 'TD2004']:
		pathtrain = Path(__file__).absolute().parents[1] / 'data' / data
		toOrdenedFile(str(pathtrain/'train.txt'), str(pathtrain/'train.dat'))

transform_ranksvm()