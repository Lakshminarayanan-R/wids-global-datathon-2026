'''
This module contains the model training and prediction functionality.
The main focus is to use the sklearns survival analysis toolkit first.
Then using tradtional methodologies to check the feature importance to further make modifications to
data transformation logic.
'''
from loguru import logger
import pandas as pd
import numpy as np
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from pydantic import ValidationError
from src.utils.wids_metrics import hybrid_score
import os

SEED = 42
CV_N_SPLITS = 5
HORIZONS_PRED = np.array([12, 24, 48, 72], dtype=float)
TRAIN_SPLIT, TEST_SPLIT = 0.80, 0.20

train_df = pd.read_csv('data/transformed/train.csv')
test_df = pd.read_csv('data/transformed/test.csv')
submission_df = pd.read_csv('data/raw/sample_submission.csv')

def get_surv_predictions(model, X):
    surv_fns = model.predict_survival_function(X)
    preds = np.empty((len(surv_fns), len(HORIZONS_PRED)), dtype=float)
    for i, fn in enumerate(surv_fns):
        t_min, t_max = fn.domain
        preds[i, :] = fn(np.clip(HORIZONS_PRED, t_min, t_max))
    return 1.0 - preds

def model_trainer(model, train_mode, dataframe, cv_n_splits = CV_N_SPLITS):
    '''
    This function is used to train the model using the historical data.
    There are 2 training modes:
    1. full - This mode takes the entire dataframe as input for training.
    2. validation - This mode splits the input data into train, validation and test.
    TO-DO: Make the validation split config to be configurable from a json.
    '''
    X = dataframe.drop(columns = ['event_id', 'event', 'time_to_hit_hours'])
    event_values = dataframe['event'].values
    time_values = dataframe['time_to_hit_hours'].values
    y = Surv.from_arrays(event=dataframe['event'].astype(bool), time=dataframe['time_to_hit_hours'])
    if train_mode == 'full':
        model.fit(X,y)
        logger.info(f"Model Successfully trained on {X.shape[0]} number of rows")
    elif train_mode == 'validation':
        X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,test_size=TEST_SPLIT, 
                                                                    random_state=SEED)
        train_val_idx = X_train_val.index.to_list()
        event_values = event_values[train_val_idx]
        time_values = time_values[train_val_idx]
        cv = StratifiedKFold(n_splits=cv_n_splits,
                             shuffle=True,
                             random_state=SEED)
        X_train_val = X_train_val.reset_index(drop=True)
        for i, (tr_idx, val_idx) in enumerate(cv.split(X_train_val, event_values)):
            model.fit(X_train_val.iloc[tr_idx], y_train_val[tr_idx])
            result = get_surv_predictions(model, X_train_val.iloc[val_idx])
            #logger.info(result)
            time_values_cv = time_values[val_idx]
            event_values_cv = event_values[val_idx]
            hybrid, ci, weighted_brier = hybrid_score(time=time_values_cv,
                                                    event=event_values_cv,
                                                    p24=result[:,1],
                                                    p48=result[:,2],
                                                    p72=result[:,3])
            logger.info(f"hybrid score: {hybrid}, "
                        f"ci value: {ci}, "
                        f"weighted brier: {weighted_brier}")
        logger.info(f"Model Successfully trained on {X_train_val.shape[0]} number of rows")
        # Set to print metrics for OOF preds (test_stage)
    else:
        raise ValidationError(f"Invalid train mode {train_mode} was given."
                              f"Acceptable train modes as ('full', 'validation')")
    return model

def submission_constructor(model, dataframe, submission_df):
    X = dataframe.drop(columns = ['event_id'])
    #logger.info(X.columns)
    results = get_surv_predictions(model, X)
    submission_df['prob_12h'] = results[:,0]
    submission_df['prob_24h'] = results[:,1]
    submission_df['prob_48h'] = results[:,2]
    submission_df['prob_72h'] = results[:,3]
    if not os.path.exists('data/submission'):
        os.mkdir('data/submission')
    submission_df.to_csv('data/submission/submission.csv', index=False)
    logger.info(f"Submission file created successfully!")
    return None

model = GradientBoostingSurvivalAnalysis()
trained_model = model_trainer(model=model, train_mode = 'full',dataframe=train_df)
submission_constructor(model, test_df, submission_df)
#logger.info(pred)
