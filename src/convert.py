
import pandas as pd
def convert(input_filename, out_data_filename)->list:
	input = open(input_filename,"r")
	output_feature = open(out_data_filename,"w")
	cur_cnt = 0
	cur_doc_cnt = 0
	last_qid = -1
	querygroup = []
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
				querygroup.append(str(cur_doc_cnt))
				cur_cnt += 1
			cur_doc_cnt = 0
			last_qid = qid
		cur_doc_cnt += 1
		output_feature.write(label+' ')
		output_feature.write(' '.join(tokens[2:]) + '\n')
	querygroup.append(str(cur_doc_cnt))
	
	input.close()
	return querygroup

def get_group_id(path):
	pd.Dataframe