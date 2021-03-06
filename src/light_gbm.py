from pathlib import Path
from convert import read_group
import lightgbm as lgb
import time
from contextlib import redirect_stdout
from params import paramsLIGHTGBM

def evaluate():
    for data in ['MSLR10K', 'MSLR30K', 'OHSUMED', 'TD2003', 'TD2004']:
        pathtrain = Path(__file__).absolute().parents[1] / 'data' / data
        train = str(pathtrain/f"{data}.train")
        test = str(pathtrain/f"{data}.test")
        vali = str(pathtrain/f"{data}.vali")
        train_group = read_group(str(pathtrain/f"{data}.train.group"))
        vali_group = read_group(str(pathtrain/f"{data}.vali.group"))

        print('Starting training...')
        
        for objective in ['regression', 'lambdarank', 'rank_xendcg']:
            lgb_train = lgb.Dataset(train, group=train_group)
            lgb_vali = lgb.Dataset(vali, reference=lgb_train, group=vali_group)
            eval_result = {}
            with open(f'train_logs/train.lgbm.{objective}.{data}.log', 'w') as f:
                with redirect_stdout(f):
                    inicio = time.time()
                    gbm = lgb.train(paramsLIGHTGBM[objective][data],
                                lgb_train,
                                valid_sets=[lgb_vali],
                                valid_names=['eval'], 
                                evals_result=eval_result,
                                )
                    fim = time.time()
                    print("Tempo de execução Total: " + str(fim - inicio) + " segundos")

            print('Saving model...')
            gbm.save_model(f'models/lightgbm.{objective}.{data}.model')

            print('Starting predicting...')

            y_pred = gbm.predict(test)
            # X_test, y_test = load_svmlight_file(test)
            # dataset = pd.DataFrame(X_test.todense())
            # dataset["label"] = y_test
            # dataset["predicted_ranking"] = y_pred
            # dataset.sort_values("predicted_ranking", ascending=False)
            ll = []
            for a in y_pred:
                ll.append(str(a))
            with open(f'predicted/lightgbm.{objective}.{data}.txt', 'w') as outfile:
                outfile.write("\n".join(ll))


evaluate()
