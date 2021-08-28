from pathlib import Path
from convert import convert

import pandas as pd
from sklearn.metrics import accuracy_score, ndcg_score
import numpy as np
import matplotlib.pyplot as plt

import lightgbm as lgb

print('Loading data...')
# load or create your dataset

regression_example_dir = Path(__file__).absolute().parents[1] / 'data/Fold1'
df_train = convert(str(regression_example_dir/'train.txt'))
df_valid = convert(str(regression_example_dir/'vali.txt'))
df_test = convert(str(regression_example_dir/'test.txt'))
#df_all = convert(str(regression_example_dir/'TD2003.txt'))

#df_train = pd.read_csv(str(regression_example_dir / 'trainingset.txt'), header=None, sep='\t')
#df_test = pd.read_csv(str(regression_example_dir / 'testset.txt'), header=None, sep='\t')

#train, validate, test = np.split(df_all.sample(frac=1, random_state=42), 
#                       [int(.6*len(df_all)), int(.8*len(df_all))])
#qids_train = train.groupby("qid")["qid"].count().to_numpy()
#qids_test = test.groupby("qid")["qid"].count().to_numpy()
#qids_valid = validate.groupby("qid")["qid"].count().to_numpy()
#y_train = train['relevance']
#y_valid = validate['relevance']
#y_test = test['relevance'].tolist()
#X_train = train.drop(['relevance', "qid"] , axis=1)
#X_valid = validate.drop(['relevance', "qid"], axis=1)
#X_test = test.drop(['relevance', "qid"], axis=1)

qids_train = df_train.groupby("qid")["qid"].count().to_numpy()
qids_test = df_test.groupby("qid")["qid"].count().to_numpy()
qids_valid = df_valid.groupby("qid")["qid"].count().to_numpy()
# qids_all = df_all.groupby("qid")["qid"].count().to_numpy()
y_train = df_train['relevance']
y_valid = df_valid['relevance']
y_test = df_test['relevance']
# y_all = df_all['relevance']
X_train = df_train.drop(['relevance', "qid"] , axis=1)
X_valid = df_valid.drop(['relevance', "qid"], axis=1)
X_test = df_test.drop(['relevance', "qid"], axis=1)
# X_all = df_all.drop(['relevance', "qid"], axis=1)

print("create dataset for lightgbm")
lgb_train = lgb.Dataset(X_train, y_train,  group=qids_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, group=qids_valid)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, group=qids_test)

# specify your configurations as a dict
params = {
    'boosting_type': 'dart',
    'objective': 'regression',
    'metric': {'ndcg', "map"},
    'eval_at': [1, 3, 5, 10],
    'learning_rate': 0.1,
    'num_leaves': 255,
    'num_trees': 500,
    'num_threads': 4,
    'min_data_in_leaf': 1,
    "tree_learner": "serial",
    'min_sum_hessian_in_leaf': 100,
    #"early_stopping_rounds":50
}

print('Starting training...')
# model = lgb.LGBMRanker(
#     objective="regression",
#     metric=['ndcg', "map"],
#     num_leaves= 255,
#     n_estimators=250,
#     learning_rate = 0.1,
#     boosting_type='gbdt',
#     num_trees = 500,
    
#     min_data_in_leaf = 0,
#     min_sum_hessian_in_leaf = 100
# )



# gbm = model.fit(
#     X=X_train,
#     y=y_train,
#     group=qids_train,
#     eval_set=[(X_valid, y_valid)],
#     eval_group=[qids_valid],
#     eval_at=[10,20],
#     eval_metric='ndcg',
#     verbose=10,
#     early_stopping_rounds=50
# )

# train
eval_result = {}
gbm = lgb.train(params,
                lgb_train,
                #num_boost_round=250,
                valid_sets=[lgb_eval, lgb_test],
                valid_names = ['eval', 'test'],
                evals_result =eval_result,
                verbose_eval=10
                )



print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')

y_pred = gbm.predict(X_test)
predictions_classes = []
for index, instance in df_test.iterrows():
    actual = instance['relevance']
    prediction = round(y_pred[index])
    predictions_classes.append(round(prediction))
    #print("actual= ", actual, ", prediction= ", prediction)

predictions_classes = np.asarray([predictions_classes])

accuracy = accuracy_score(predictions_classes[0], y_test)*100
print(accuracy,"%")

true_relevance = np.asarray([y_test.tolist()])
ndcg1 = ndcg_score(true_relevance, predictions_classes, k=1)
print('NDCG@1', ndcg1)
ndcg3 = ndcg_score(true_relevance, predictions_classes, k=3)
print('NDCG@3', ndcg3)
ndcg5 = ndcg_score(true_relevance, predictions_classes, k=5)
print('NDCG@5', ndcg5)
ndcg10 = ndcg_score(true_relevance, predictions_classes, k=10)
print('NDCG@10', ndcg10)


lgb.plot_importance(gbm, max_num_features = 50)
plt.show()

lgb.plot_tree(gbm)
plt.show()

lgb.plot_metric(eval_result, 'ndcg')
plt.show()

lgb.plot_metric(eval_result, 'map')
plt.show()


print('fim')