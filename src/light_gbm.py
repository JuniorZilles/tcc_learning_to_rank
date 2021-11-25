from pathlib import Path
from convert import read_group
import matplotlib.pyplot as plt

import lightgbm as lgb
from contextlib import redirect_stdout
from params import paramsLIGHTGBM

def evaluate():
    for data in ['MSLR10K', 'MSLR30K', 'OHSUMED', 'TD2003', 'TD2004']:
        pathtrain = Path(__file__).absolute().parents[1] / 'data' / data
        train = str(pathtrain/f"{data}.train")
        test = str(pathtrain/f"{data}.test")
        vali = str(pathtrain/f"{data}.vali")
        train_group = read_group(str(pathtrain/f"{data}.train.group"))
        test_group = read_group(str(pathtrain/f"{data}.test.group"))
        vali_group = read_group(str(pathtrain/f"{data}.vali.group"))

        lgb_train = lgb.Dataset(train, group=train_group)
        lgb_test = lgb.Dataset(test, reference=lgb_train, group=test_group)
        lgb_vali = lgb.Dataset(vali, reference=lgb_train, group=vali_group, free_raw_data = False)


        # specify your configurations as a dict
        print('Starting training...')

        # train
        for objective in ['lambdarank', 'rank_xendcg']:
            eval_result = {}
            with open(f'train.lgbm.{objective}.{data}.log', 'w') as f:
                paramsLIGHTGBM['objective'] = objective
                with redirect_stdout(f):
                    gbm = lgb.train(paramsLIGHTGBM[data],
                                lgb_train,
                                valid_sets=[lgb_test],
                                valid_names=['eval'], 
                                evals_result=eval_result,
                                )

            print('Saving model...')
            # save model to file
            gbm.save_model(f'lightgbm.{objective}.{data}.model')

            print('Starting predicting...')

            y_pred = gbm.predict(vali, num_iteration=gbm.best_iteration)
            # eval
            lgb_vali.construct()
            dataset = lgb_vali.get_data()
            dataset["predicted_ranking"] = y_pred
            dataset.sort_values("predicted_ranking", ascending=False)
            dataset.to_csv(f'lightgbm.{objective}.{data}.vali.predicted.csv')

            lgb.plot_importance(gbm, max_num_features=50)
            #plt.show()
            plt.savefig(f'train.lgbm.{objective}.{data}.importance.png', dpi=1920, orientation='portrait')

            lgb.plot_tree(gbm)
            #plt.show()
            plt.savefig(f'train.lgbm.{objective}.{data}.tree.png', dpi=1920, orientation='portrait')

            print('fim')

evaluate()
