svm_rank_learn -c 0.1 data/OHSUMED/train.txt models/rankSVM.OHSUMED.model 1> train_logs/train.rankSVM.OHSUMED.log 2>&1 

svm_rank_classify data/OHSUMED/test.txt models/rankSVM.OHSUMED.model predicted/rankSVM.OHSUMED.txt

svm_rank_learn -c 0.1 data/TD2003/train.txt models/rankSVM.TD2003.model 1> train_logs/train.rankSVM.TD2003.log 2>&1

svm_rank_classify data/TD2003/test.txt models/rankSVM.TD2003.model predicted/rankSVM.TD2003.txt 