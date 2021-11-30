
from pathlib import Path
from convert import toDataframe
from flaml import AutoML

for data in ['MSLR10K', 'MSLR30K', 'OHSUMED', 'TD2003', 'TD2004']:
    path = Path(__file__).absolute().parents[1] / 'data' / data
    group_train, y_train, X_train = toDataframe(str(path/'train.txt'))
    for task in ['regression', 'rank']:
        for boost in ['lgbm', 'xgboost']:
                automl = AutoML()
                automl.fit(
                    X_train, y_train, groups=group_train, log_file_name=f'flaml_logs/{task}.{boost}.{data}.flaml.log', 
                    task=task, time_budget=10000,  estimator_list=[boost], 
                )
                # Print the best model
                with open(f'flaml_logs/{task}.{boost}.{data}.flaml.log', 'a') as f:
                    f.write(f'Best ML leaner: {automl.best_estimator}')
                    f.write(f'Best hyperparmeter config: {automl.best_config}')
                    f.write(str(automl.model.estimator))

