import xgboost as xgb
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_svmlight_file
import time
from contextlib import redirect_stdout
from params import paramsXGBOOST

def evaluate_xgboost():
    for data in ['MSLR10K', 'MSLR30K', 'OHSUMED', 'TD2003', 'TD2004']:
        path = Path(__file__).absolute().parents[1] / 'data' / data
        test = str(path/f"{data}.test")

        print("Carregando arquivos")
        dtrain = xgb.DMatrix(str(path/'train.txt'))
        dvali = xgb.DMatrix(str(path/'vali.txt'))
        dtest = xgb.DMatrix(str(path/'test.txt'))

        eval = [(dvali, 'eval')]

        print('Treinando')
        for objective in ['regression', 'rank']:
            with open(f'train_logs/train.xgboost.{objective}.{data}.log', 'w') as f:
                with redirect_stdout(f):
                    inicio = time.time()
                    bst = xgb.train(paramsXGBOOST[objective][data], dtrain, 500, evals=eval)
                    fim = time.time()
                    print("Tempo de execução Total: " + str(fim - inicio) + " segundos")
            print('Salvando o modelo')
            bst.save_model(f'models/xgboost.{objective}.{data}.model')

            print('Predizendo')
            y_pred = bst.predict(dtest)
            X_test, y_test = load_svmlight_file(test)
            dataset = pd.DataFrame(X_test.todense())
            dataset["label"] = y_test
            dataset["predicted_ranking"] = y_pred
            #dataset.sort_values("predicted_ranking", ascending=False)
            dataset.to_csv(f'predicted_csv/xgboost.{objective}.{data}.test.predicted.csv')



evaluate_xgboost()