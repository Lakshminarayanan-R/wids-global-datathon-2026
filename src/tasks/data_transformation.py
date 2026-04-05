'''
This module contains functions for data transformation tasks, 
including scaling, encoding, and feature engineering. 
It provides utilities to preprocess data before feeding it 
into machine learning models. 
The functions are designed to be flexible and can handle 
various types of data, such as numerical and categorical 
features.
NOTE: Preserve the order of the rows in the dataframe while performing transformations, since the order of the rows is important for the submission.
'''

from typing import Literal
import json
import os
import pandas as pd
import numpy as np
from loguru import logger
from argparse import ArgumentParser
from pydantic import ValidationError

ID_VARIABLE = 'event_id'
RAW_TARGET = ['event', 'time_to_hit_hours']
FEATURES_WITH_HIGH_NOISE = ['event_start_hour', 'event_start_dayofweek'] 
HIGHLY_REDUNDANT_COLS = ['area_first_ha','area_growth_abs_0_5h','spread_bearing_deg']
STRAT_THR = 5000
EXCEPTIONAL_COLS = ['closing_speed_m_per_h']

def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates new features based on the existing features in the DataFrame.
    The new features are created based on the notebook - `https://www.kaggle.com/code/dasdasdada/0-97124-gbsa-rsf-lgb-survival-engine`.
    For example, we can create new features by combining existing features or by applying mathematical operations on them.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame from which new features will be created.
    
    Returns:
    pd.DataFrame: The DataFrame after adding new features.
    """
    # Example of creating a new feature by combining existing features
    result = df.copy()
    dist = result['dist_min_ci_0_5h'].clip(lower=1)
    speed = result['closing_speed_m_per_h']
    perimeters = result['num_perimeters_0_5h']
    area_first = result['area_first_ha']
    result['log_distance'] = np.log1p(dist)
    result['inv_distance'] = 1 / (dist / 1000 + 0.1)
    result['inv_distance_sq'] = result['inv_distance'] ** 2
    result['sqrt_distance'] = np.sqrt(dist)
    result['dist_km'] = dist / 1000
    result['dist_km_sq'] = (dist / 1000) ** 2
    result['dist_rank'] = dist.rank(pct=True)
    fire_radius = np.sqrt(area_first * 10000 / np.pi)
    result['fire_radius_km'] = fire_radius / 1000
    result['radius_to_dist'] = fire_radius / dist
    result['area_to_dist_ratio'] = area_first / (dist / 1000 + 0.1)
    result['log_area_dist_ratio'] = np.log1p(area_first) - np.log1p(dist)
    result['has_movement'] = (perimeters > 1).astype(float)
    closing_pos = speed.clip(lower=0)
    result['eta_hours'] = np.where(closing_pos > 0.01, dist / closing_pos, 9999).clip(max=9999)
    result['log_eta'] = np.log1p(result['eta_hours'].clip(0, 9999))
    radial_growth = result['radial_growth_rate_m_per_h'].clip(lower=0)
    effective_closing = closing_pos + radial_growth
    result['effective_closing_speed'] = effective_closing
    result['eta_effective'] = np.where(effective_closing > 0.01, dist / effective_closing, 9999).clip(max=9999)
    result['threat_score'] = result['alignment_abs'] * speed / np.log1p(dist)
    result['fire_urgency'] = perimeters * speed
    result['growth_intensity'] = result['area_growth_rate_ha_per_h'] * perimeters
    result['zone_near'] = (dist < STRAT_THR).astype(float)
    result['zone_far'] = (dist >= STRAT_THR).astype(float)
    result['is_summer'] = result['event_start_month'].isin([6, 7, 8]).astype(float)
    # result['is_afternoon'] = ((result['event_start_hour'] >= 12) & (result['event_start_hour'] < 20)).astype(float)
    # Zone-specific ranks
    near_mask = dist < STRAT_THR
    result['near_speed_rank'] = 0.0
    if near_mask.sum() > 0:
        result.loc[near_mask, 'near_speed_rank'] = speed[near_mask].rank(pct=True)
    result['far_threat_rank'] = 0.0
    far_mask = ~near_mask
    if far_mask.sum() > 0:
        result.loc[far_mask, 'far_threat_rank'] = result.loc[far_mask, 'threat_score'].rank(pct=True)
    drop_cols = ['relative_growth_0_5h', 'projected_advance_m', 'centroid_displacement_m',
                 'centroid_speed_m_per_h', 'closing_speed_abs_m_per_h', 'area_growth_abs_0_5h']
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    logger.info(f"Total features after feature engineering {result.shape[1]}")
    #logger.info(f"{result.columns}")
    return result

def remove_high_noise_features(
        df: pd.DataFrame,
        features_with_high_noise: list[str] = FEATURES_WITH_HIGH_NOISE) -> pd.DataFrame:
    """
    This function removes features that cause high noise to the data.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame from which high noise features will be removed.
    
    Returns:
    pd.DataFrame: The DataFrame after removing high noise features.
    """
    return df.drop(columns=features_with_high_noise)

def removing_features_with_high_correlation(
        df: pd.DataFrame,
        correlation_threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    """
    This function identifies the features that have high correlation with other features and removes them from the DataFrame.
    The logic for identifying features with high correlation is as follows:
    1. Calculate the correlation matrix of the dataframe with all features except the ID and target variables.
    2. Calculate the correlation of each feature with the target variables (now the target is `time_to_hit`).
    3. If the correlation of a feature is greater than the correlation threshold:
        a. Check if the feature has high correlation with any other feature (correlation > 0.95).
        b. If it does, remove the feature with the lower correlation with the target variable.
    """
    # Removing the id column from the dataframe for correlation calculation
    dataframe_for_correlation_without_id_and_target = df.drop(columns=[ID_VARIABLE] + RAW_TARGET)
    dataframe_for_correlation_without_id = df.drop(columns=[ID_VARIABLE])
    # Step 1: Calculate the correlation matrix of the dataframe with all features except the ID and target variables.
    corr_matrix = dataframe_for_correlation_without_id_and_target.corr().abs()
    upper_triangle = dataframe_for_correlation_without_id_and_target.corr().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = upper_triangle.stack().reset_index()
    high_corr.columns = ['Feature1', 'Feature2', 'Correlation_between_f1_f2']

    # Step 2: Calculate the correlation of each feature with the target variable.
    target_correlation_with_event = dataframe_for_correlation_without_id.corr()[['event']].abs().sort_values(by='event', ascending=False)
    target_correlation_with_time_to_hit_hrs = dataframe_for_correlation_without_id.corr()[['time_to_hit_hours']].abs().sort_values(by='time_to_hit_hours', ascending=False)
    
    # Step 2.1: Constructing a correlation factor matrix that contains the correlation of each feature with the target variable
    event_corr_with_feature_1 = target_correlation_with_event.reset_index()
    event_corr_with_feature_1.rename(columns={'index': 'Feature1', 'event': 'event_corr_f1'}, inplace = True)
    event_corr_with_feature_2 = target_correlation_with_event.reset_index()
    event_corr_with_feature_2.rename(columns={'index': 'Feature2', 'event': 'event_corr_f2'}, inplace = True)
    time_to_hit_corr_with_feature_1 = target_correlation_with_time_to_hit_hrs.reset_index()
    time_to_hit_corr_with_feature_1.rename(columns={'index': 'Feature1', 'time_to_hit_hours': 'time_to_hit_corr_f1'}, inplace = True)
    time_to_hit_corr_with_feature_2 = target_correlation_with_time_to_hit_hrs.reset_index()
    time_to_hit_corr_with_feature_2.rename(columns={'index': 'Feature2', 'time_to_hit_hours': 'time_to_hit_corr_f2'}, inplace = True)

    # Step 3: Constructing the final correlation matrix
    # The goal is to identify highly correlated features (greater than correlation threshold) so removing features with low correlation factor for this scenario.
    final_dataframe = high_corr[high_corr['Correlation_between_f1_f2'] > correlation_threshold]
    final_dataframe = final_dataframe.sort_values(by='Correlation_between_f1_f2', ascending=False)
    final_dataframe = final_dataframe.merge(event_corr_with_feature_1, on='Feature1', how='left')
    final_dataframe = final_dataframe.merge(event_corr_with_feature_2, on='Feature2', how='left')
    final_dataframe = final_dataframe.merge(time_to_hit_corr_with_feature_1, on='Feature1', how='left')
    final_dataframe = final_dataframe.merge(time_to_hit_corr_with_feature_2, on='Feature2', how='left')

    # Step 3.1: Removing features with high correlation value within themselves (`Correlation_between_f1_f2`) and low correlation with the target variable (`event_corr_f1`, `event_corr_f2`, `time_to_hit_corr_f1`, `time_to_hit_corr_f2`).
    features_to_remove = set()
    for row in final_dataframe.itertuples():
        if row.Correlation_between_f1_f2 > correlation_threshold:
            if abs(row.event_corr_f1) + abs(row.time_to_hit_corr_f1) < abs(row.event_corr_f2) + abs(row.time_to_hit_corr_f2):
                features_to_remove.add(row.Feature1)
            else:
                features_to_remove.add(row.Feature2)
    logger.info(f"Identified {len(features_to_remove)} features to remove due to high correlation: {features_to_remove}")
    features_to_remove.remove(EXCEPTIONAL_COLS[0])
    logger.info(f"Removed exceptional columns from the list of features to remove: {type(EXCEPTIONAL_COLS)}")
    with open("data/transformed/features_to_remove.json","w") as f:
        json.dump(list(features_to_remove), f)
    return df.drop(columns=list(features_to_remove))

def removing_features_with_high_correlation_with_config(dataframe) -> pd.DataFrame:
    '''
    Check if the config file exists and remove the highly correlated features from the dataframe.
    This function has a dependency of the data transformation to be run with the training data atleast once.
    '''
    try:
        if os.path.exists('data/transformed/features_to_remove.json'):
            with open('data/transformed/features_to_remove.json', "r") as f:
                features_to_remove = json.load(f)
            uncorrelated_df = dataframe.drop(columns=features_to_remove)
            return uncorrelated_df
    except FileNotFoundError:
        print("The config file to remove highly correlated features not found.")

def transform_data() -> pd.DataFrame:
    """
    This is the main function for transforming the data before modeling.
    Following are the transformation logics performed on the data:
    1. Identifying and removing features that cause high noise to the data.
    2. Removing features that has high correlation with other features (correlation > 0.95).
    3. Performing feature engineering to create new features that may be useful for modeling.
    4. Creating X and y datasets for modeling. This dataset has 4 target variables (y1, y2, y3, y4) and the rest of the features are used as input features (X).
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to be transformed.
    
    Returns:
    pd.DataFrame: The transformed DataFrame ready for modeling.
    """
    parser = ArgumentParser(description="Data Transformation")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--dataframe-name", type=str, required=True, help="Name of the dataframe (train/test)")
    parser.add_argument("--correlation-threshold", type=float, default=0.95, help="Threshold for identifying highly correlated features")
    args = parser.parse_args()
    logger.info("Starting data transformation...")
    #logger.info(f"{args}")

    # Step 0: Load the data to a dataframe
    df = pd.read_csv(args.input_file)

    # Step 1: Remove features with high noise
    denoised_df = remove_high_noise_features(df, FEATURES_WITH_HIGH_NOISE)
    logger.info("Removed features with high noise: {}", FEATURES_WITH_HIGH_NOISE)
    logger.info(f"Dataframe shape after removing high noise features: {denoised_df.shape}")

    # Step 2: Remove features with high correlation
    if args.dataframe_name == 'train':
        uncorrelated_df = removing_features_with_high_correlation(denoised_df, 
                                                              correlation_threshold=0.95)
    elif args.dataframe_name == 'test':
        if os.path.exists('data/transformed/features_to_remove.json'):
            uncorrelated_df = removing_features_with_high_correlation_with_config(denoised_df)

    else:
        raise ValidationError(f"{args.dataframe_name} is not supported, Should be one of train / test...")
    logger.info(f"Dataframe shape after removing highly correlated features: {uncorrelated_df.shape}")

    # Step 3: Perform feature engineering to create new features that may be useful for modeling.
    engineered_df = create_new_features(uncorrelated_df)

    # Step 4: Create X and y datasets for modeling. This dataset has 4 target variables (y1, y2, y3, y4) and the rest of the features are used as input features (X).
    # if args.dataframe_name == 'train':
    #     engineered_df['prob_12h'] = np.where((engineered_df['time_to_hit_hours'] <= 12) & (engineered_df['event'] == 1), 1, 0)
    #     engineered_df['prob_24h'] = np.where((engineered_df['time_to_hit_hours'] <= 12) & (engineered_df['event'] == 1), 1, 0)
    #     engineered_df['prob_48h'] = np.where((engineered_df['time_to_hit_hours'] <= 12) & (engineered_df['event'] == 1), 1, 0)
    #     engineered_df['prob_72h'] = np.where((engineered_df['time_to_hit_hours'] <= 12) & (engineered_df['event'] == 1), 1, 0)
    logger.info("Completed data transformation.")
    logger.info(f"Final dataframe shape after transformation: {engineered_df.shape}")
    logger.info(f"column names after transformation: {engineered_df.columns.tolist()}")

    # Saving the final dataframe as csv
    engineered_df.to_csv(f"data/transformed/{args.dataframe_name}.csv")
    return df

if __name__ == "__main__":
    transform_data()