
from pathlib import Path
from convert import toDataframe
from flaml import AutoML
#'MSLR10K', 'MSLR30K',
for data in ['OHSUMED', 'TD2003', 'TD2004']:
    path = Path(__file__).absolute().parents[1] / 'data' / data
    group_train, y_train, X_train = toDataframe(str(path/'train.txt'))
    group_vali, y_vali, X_vali = toDataframe(str(path/'vali.txt'))
    group_test, y_test, X_test = toDataframe(str(path/'test.txt'))
    for task in ['rank', 'regression', ]:
        for boost in ['lgbm', 'xgboost']:
            if data == 'OHSUMED' and task == 'rank':
                continue
            elif data == 'OHSUMED' and task == 'regression' and boost == 'lgbm':
                continue
            else:
                automl = AutoML()
                automl.fit(
                    X_train=X_train, y_train=y_train, X_val=X_vali, y_val=y_vali, groups=group_train, groups_val=group_vali, log_file_name=f'flaml_logs/{task}.{boost}.{data}.flaml.log',
                    estimator_list=[boost], early_stop=True, time_budget=10000, task=task,
                    
                )
                automl.predict(X_test)
                # Print the best model
                with open(f'flaml_logs/{task}.{boost}.{data}.flaml.log', 'a') as f:
                    f.write(f'Best ML leaner: {automl.best_estimator}')
                    f.write(f'Best hyperparmeter config: {automl.best_config}')
                    f.write(str(automl.model.estimator))
