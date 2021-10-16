import xgboost as xgb
from pathlib import Path
from contextlib import redirect_stdout

def evaluate_xgboost():

    path = Path(__file__).absolute().parents[1] / 'data/Fold1'

    print("Carregando arquivos")
    dtrain = xgb.DMatrix(str(path/'train.txt'))
    dvali = xgb.DMatrix(str(path/'test.txt'))

    evallist = [(dvali, 'eval'), (dtrain, 'train')]


    param = {"eta": 0.1,
            "max_depth": 8,
            
            "min_child_weight": 100,
            "nthread": 6,
            "gamma": 0,
            "lambda": 0,
            "alpha": 0,
            "verbosity": 2,
            "tree_method": "exact",
            'objective': 'rank:ndcg',
            "task":"train",
            'eval_metric': ["map@1", "map@3", "map@5", "map@10", 'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10']}

    print('Treinando')
    num_round = 500
    bst = xgb.train(param, dtrain, num_round, evallist)

    print('Salvando o modelo')
    bst.save_model('xgboost.model')

    print('Predizendo')
    dtest = xgb.DMatrix(str(path/'test.txt'))

    ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))

    print('Plotando as importancias das features')
    xgb.plot_importance(bst)

    print('Plotando 2 das arvores')
    xgb.plot_tree(bst, num_trees=2)


#with open('out_xgboost.txt', 'w') as f:
#    with redirect_stdout(f):
evaluate_xgboost()