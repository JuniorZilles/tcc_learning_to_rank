from pathlib import Path
from convert import convert
import matplotlib.pyplot as plt

import lightgbm as lgb
from contextlib import redirect_stdout

def evaluate():
    print('Loading data...')
    path = Path(__file__).absolute().parents[1] / 'data/Fold1' #ndcg@10 -> 0.52..
    pathtrain = Path(__file__).absolute().parents[1] / 'data/train'
    train = str(pathtrain/"mslr.train")
    test = str(pathtrain/"mslr.test")
    vali = str(pathtrain/"mslr.vali")
    group_train = convert(str(path/'train.txt'),train)
    group_test = convert(str(path/'test.txt'),test)
    group_vali = convert(str(path/'test.txt'),vali)

    #df_train = convert(str(path/'train.txt'))
    #df_valid = convert(str(regression_example_dir/'vali.txt'))
    #df_test = convert(str(path/'test.txt'))



    # qids_train = df_train.groupby("qid")["qid"].count().to_numpy()
    # qids_test = df_test.groupby("qid")["qid"].count().to_numpy()
    # y_train = df_train['relevance']
    # y_test = df_test['relevance']
    # X_train = df_train.drop(['relevance', "qid"] , axis=1)
    # X_test = df_test.drop(['relevance', "qid"], axis=1)


    print("create dataset for lightgbm")
    
    lgb_train = lgb.Dataset(train, group=group_train)
    #lgb_train = lgb.Dataset(X_train, y_train,  group=qids_train)
    #lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, group=qids_valid)
    lgb_eval = lgb.Dataset(vali, reference=lgb_train, group=group_vali)
    #lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, group=qids_test)
    lgb_test = lgb.Dataset(test, reference=lgb_train, group=group_test)
    #lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, group=qids_test)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': {'ndcg', "map"},
        'eval_at': [1, 3, 5, 10],
        "max_bin" : 255, #max number of bins that feature values will be bucketed in
        'learning_rate': 0.1,
        'num_leaves': 255,
        "num_iterations": 500,
        'num_threads': 6,
        'min_data_in_leaf': 1,
        "task":"train",
        "tree_learner": "serial",
        'min_sum_hessian_in_leaf': 100,
        #"early_stopping_rounds":50
    }

    print('Starting training...')

    # train
    eval_result = {}
    gbm = lgb.train(params,
                    lgb_train,
                    #num_boost_round=250,
                    valid_sets=[ lgb_test],#lgb_eval
                    valid_names = ['eval'],#'eval'
                    evals_result =eval_result,
                    #verbose_eval=10
                    )



    print('Saving model...')
    # save model to file
    gbm.save_model('lightgbm.model')

    print('Starting predicting...')

    y_pred = gbm.predict(vali)
    # predictions_classes = []
    # for index, instance in df_test.iterrows():
    #     actual = instance['relevance']
    #     prediction = round(y_pred[index])
    #     predictions_classes.append(round(prediction))
    #     #print("actual= ", actual, ", prediction= ", prediction)

    # predictions_classes = np.asarray([predictions_classes])

    # accuracy = accuracy_score(predictions_classes[0], y_test)*100
    # print(accuracy,"%")

    # true_relevance = np.asarray([y_test.tolist()])



    lgb.plot_importance(gbm, max_num_features = 50)
    plt.show()

    lgb.plot_tree(gbm)
    plt.show()



    print('fim')

#with open('out.txt', 'w') as f:
#    with redirect_stdout(f):
evaluate()