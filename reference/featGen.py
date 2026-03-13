import numpy as np
import os
import pandas as pd
import glob
import pickle
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import platform
import re

import physio_processing as pp
from heartview.pipeline import PPG, EDA

def dictval2str(dict, n_vals=2):
    s = ''
    for dict_val in list(dict.values())[0:n_vals]:
        s += '_' + str(dict_val)
    return s

def list_folders_in_dir(dir):
    # list all folders including the current path
    folders = [x[0] for x in os.walk(dir)]
    # delete current path from list
    # del folders[0]
    return folders[1:]

#uIDDict = getUserIdsDictionaryFromDirectoryList(dirList, pathStyle, idNDigits):
# where uIDDict is a dictionary {newID,UID} where UID is a list with the
# different unique user codes, and newID is a contiguous positive integer number
# (1 to NumberOfUsers) associated with each UID.
# Inputs: dirList a list containing all directory names;
# pathStyle: that should be '/' for linux style or '\' for windows.
# idNDigits: this code assumes that each user ID is identified by the
# first idNDigits of the folders containing the csv data files.
def get_uid_dict_from_dir_list(dir_list, path_style, id_ndigits=4):
    # ids = [x.replace(path + path_style, '') for x in folders]
    uid = [x.split(path_style)[-1][0:id_ndigits] for x in dir_list]
    new_id = np.unique(uid, return_inverse=True)[1]
    return dict(zip(new_id + 1, uid))


def get_uid_dict_from_dir(dir, path_style, id_ndigits=4):
    return get_uid_dict_from_dir_list(list_folders_in_dir(dir), path_style, id_ndigits)

def generate_raw_dataset():
    pass

def select_feat_from_feat_code(feat_code, o_is_new_dataset=False):
    selected_feat = []

    #There is no IBI for the new dataset!!
    if o_is_new_dataset:
        acc_signals = ['ACC_X', 'ACC_Y', 'ACC_Z']
        other_signals = ['EDA', 'BVP']
        other_signals_filtered = ['EDA_filtered', 'BVP_filtered']
        derived_signals = ['HR', 'RMSSD', 'TONIC', 'PHASIC']
        if feat_code == 4:
            print('Error: feat_code cannot be 4 for this dataset!')
            exit()
    else:
        acc_signals = ['X', 'Y', 'Z']
        other_signals = ['EDA', 'BVP', 'IBI']
        other_signals_filtered = ['EDA_filtered', 'BVP_filtered', 'IBI']

    feats = ['first', 'last', 'max', 'min', 'mean', 'median', 'nunique', 'std', 'sum', 'var']
    other_feat = ['AGGObserved', 'TimePastAggression']

    if feat_code == -1:
        selected_feat = ['TimePastAggression']

    elif feat_code == 0:
        selected_feat = ['AGGObserved', 'TimePastAggression']

    elif feat_code == 1:
        # ACC only
        selected_feat = [acc_signals[i] + feats[j] for i in range(len(acc_signals)) for j in range(len(feats))]

    elif feat_code == 2:
        # EDA only
        signals = other_signals[0]
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]

    elif feat_code == 3:
        #BVP only
        signals = other_signals[1]
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]

    elif feat_code == 4:
        #IBI only (not present for the new dataset)
        signals = other_signals[2]
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]

    elif feat_code == 5:
        #BVP + EDA
        signals = other_signals[0:2]
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]

    elif feat_code == 6:
        # acc_signals + other_signals
        signals = acc_signals + other_signals
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]

    elif feat_code == 7:
        # acc_signals + other_signals
        signals = acc_signals + other_signals
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat + other_feat

    elif feat_code == 8:
        print("feat code 8")
        selected_feat = ['ACCNormmean']

    elif feat_code == 9:
        # BVP + EDA + 'AGGObserved', 'TimePastAggression'
        signals = other_signals
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat + other_feat

    elif feat_code == 10:
        # ACC + EDA + 'AGGObserved', 'TimePastAggression'
        signals = acc_signals + [other_signals[0]]
        print(signals)
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat + other_feat

    elif feat_code == 11:
        # ACC + BVP + 'AGGObserved', 'TimePastAggression'
        signals = acc_signals + [other_signals[1]]
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat + other_feat

    elif feat_code == 12:
        #'AGGObserved', 'TimePastAggression'
        selected_feat = other_feat

    elif feat_code == 13:
        # 'ACC', 'AGGObserved', 'TimePastAggression'
        selected_feat = other_feat + [acc_signals[i] + feats[j] for i in range(len(acc_signals)) for j in range(len(feats))]
        print("feat code 13:", selected_feat)
    
    elif feat_code == 14:
        # all signals (ACC+EDA+BVP+HR+RMSSD+TONIC+PHASIC)
        signals = acc_signals + other_signals + derived_signals
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]

    elif feat_code == 15:
        # all signals (ACC+EDA+BVP+HR+RMSSD+TONIC+PHASIC) + ['AGGObserved', 'TimePastAggression']
        signals = acc_signals + other_signals + derived_signals
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat + other_feat

    elif feat_code == 16:
        # all signals with filtered physio (ACC+EDA_filtered+BVP_filtered+HR+RMSSD+TONIC+PHASIC) + ['AGGObserved', 'TimePastAggression']
        signals = acc_signals + other_signals_filtered + derived_signals
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat + other_feat
    
    elif feat_code == 17:
        # ACC+EDA_filtered+BVP_filtered+ ['AGGObserved', 'TimePastAggression']
        signals = acc_signals + other_signals_filtered
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat + other_feat
    
    elif feat_code == 18: 
        # all signals with filtered physio (ACC+EDA_filtered+BVP_filtered+HR+RMSSD+TONIC+PHASIC)
        signals = acc_signals + other_signals_filtered + derived_signals
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat
    
    elif feat_code == 19:
        # ACC+EDA_filtered+BVP_filtered
        signals = acc_signals + other_signals_filtered
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat
    
    elif feat_code == 20:
        # ACC+HR+RMSSD+TONIC+PHASIC+ ['AGGObserved', 'TimePastAggression']
        signals = acc_signals + derived_signals
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat + other_feat
    
    elif feat_code == 21:
        # ACC+HR+RMSSD+TONIC+PHASIC
        signals = acc_signals + derived_signals
        selected_feat = [signals[i] + feats[j] for i in range(len(signals)) for j in range(len(feats))]
        selected_feat = selected_feat

    else:       # Including and beyond useFeaturesCode == 7
        print('Not implemented yet!')
    return selected_feat

def match_condition_with_timestamp(df, timestamps):
    """
    Based on the dataframe with matching condition column, 
    match the condition with the given timestamp array by choosing the closest timestamp's condition.
    Dataframe's index should be the timestamp.
    """
    # Convert timestamps to pandas Series for vectorized operations
    target_timestamps = pd.Series(timestamps)
    
    # Use searchsorted to find the closest indices efficiently
    indices = df.index.searchsorted(target_timestamps)
    
    # Handle edge cases where indices might be out of bounds
    indices = np.clip(indices, 0, len(df.index) - 1)
    
    # Vectorized check for whether previous index is closer
    prev_indices = np.maximum(indices - 1, 0)
    current_distances = np.abs(df.index[indices] - target_timestamps)
    prev_distances = np.abs(df.index[prev_indices] - target_timestamps)
    
    # Use previous index if it's closer
    final_indices = np.where(prev_distances < current_distances, prev_indices, indices)
    
    return df.iloc[final_indices]['Condition'].tolist()

def filter_bvp(df, low=0.5, high=4.0, order=4, overwrite_bvp=False):
    """
    Filter the BVP signal using a Chebyshev Type II filter.
    """
    df = df.copy()
    bvp_col = 'BVP'
    if not bvp_col in df.columns:
        raise ValueError(f"Column {bvp_col} not found in dataframe")
    record_duration = (df.index[-1] - df.index[0]).total_seconds()
    record_length = len(df)
    fs = round(record_length / record_duration, 0)
    df[bvp_col] = df[bvp_col].astype(float)
    # Filter BVP signal
    filtered_bvp = pp.filter_ppg(df[bvp_col].reset_index(drop=True), fs, low=low, high=high, order=order)
    if overwrite_bvp:
        df[bvp_col] = filtered_bvp
    else:
        df[bvp_col + '_filtered'] = filtered_bvp
    return df

def filter_eda(df, overwrite_eda=False):
    df = df.copy()
    eda_col = 'EDA'
    if not eda_col in df.columns:
        raise ValueError(f"Column {eda_col} not found in dataframe")
    record_duration = (df.index[-1] - df.index[0]).total_seconds()
    record_length = len(df)
    fs = round(record_length / record_duration, 0)
    eda_filters = EDA.Filters(fs=fs)
    df[eda_col] = df[eda_col].astype(float)
    if overwrite_eda:
        df[eda_col] = eda_filters.gaussian(df[eda_col])
    else:
        df[eda_col + '_filtered'] = eda_filters.gaussian(df[eda_col])
    return df

def gen_ppg_features(df):
    """
    Generate dataframes for HR and RMSSD with condition matching the timestamp.
    """
    df = df.copy()
    bvp_col = 'BVP_filtered'
    if not bvp_col in df.columns:
        raise ValueError(f"Column {bvp_col} not found in dataframe")

    record_duration = (df.index[-1] - df.index[0]).total_seconds()
    record_length = len(df)
    fs = round(record_length / record_duration, 0)

    timestamps = df.index.tolist()
    conditions = df['Condition'].tolist()
    # Detect heartbeats
    BeatDetector = PPG.BeatDetectors(fs=fs, preprocessed=True)
    df[bvp_col] = df[bvp_col].astype(float)
    beat_idx = BeatDetector.adaptive_threshold(df[bvp_col])
    # Calculate instantaneous heart rate
    instantaneous_heart_rate = pp.get_instantaneous_heart_rate(beat_idx, fs)
    # Calculate RMSSD with sliding window
    window_size = 30
    window_step = 1
    rmssd = pp.get_rmssd(beat_idx, fs, window_size=window_size, window_step=window_step)
    n_windows = len(rmssd)
    # Create DataFrames for HR and RMSSD
    df_hr = pd.DataFrame({
        'Timestamp': [timestamps[i] for i in beat_idx[1:]], 
        'HR': instantaneous_heart_rate,
        'Condition': [conditions[i] for i in beat_idx[1:]]
    })
    df_hr = df_hr.set_index('Timestamp')

    timestamps_rmssd = [
        timestamps[beat_idx[1]] + pd.Timedelta(seconds=window_step * i + window_size) for i in range(n_windows)
    ]
    conditions_rmssd = match_condition_with_timestamp(df, timestamps_rmssd)
    df_rmssd = pd.DataFrame({
        'Timestamp': timestamps_rmssd, 
        'RMSSD': rmssd,
        'Condition': conditions_rmssd
    })
    df_rmssd = df_rmssd.set_index('Timestamp')
    return df_hr, df_rmssd

def gen_eda_features(df):
    df = df.copy()
    eda_col = 'EDA_filtered'
    if not eda_col in df.columns:
        raise ValueError(f"Column {eda_col} not found in dataframe")
    record_duration = (df.index[-1] - df.index[0]).total_seconds()
    record_length = len(df)
    fs = round(record_length / record_duration, 0)
    df[eda_col] = df[eda_col].astype(float)
    phasic, tonic = EDA.get_phasic_tonic(df[eda_col], fs=fs)
    df_phasic_tonic = pd.DataFrame({
        'Timestamp': df.index,
        'PHASIC': phasic,
        'TONIC': tonic,
        'Condition': df['Condition']
    })
    df_phasic_tonic = df_phasic_tonic.set_index('Timestamp')
    return df_phasic_tonic

def gen_stats_features(df, evidence, binFeat, binSize, binLabel):
    binFeat[evidence + 'first'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).first()
    binFeat[evidence + 'last'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).last()
    binFeat[evidence + 'max'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).max()
    binFeat[evidence + 'min'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).min()
    binFeat[evidence + 'mean'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).mean()
    binFeat[evidence + 'median'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).median()
    binFeat[evidence + 'nunique'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).nunique()
    binFeat[evidence + 'std'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).std()
    binFeat[evidence + 'sum'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).sum()
    binFeat[evidence + 'var'] = pd.to_numeric(df[evidence].dropna()).resample(binSize).var()
    binLabel[evidence + 'Label'] = df.Condition.astype(float).resample(binSize).max()
    return binFeat, binLabel

def feat_generator(inputDict, binSize, aggCategory, nonAggCategory, onset, o_is_new_dataset=False):
    listAllInstances = inputDict['dataAll']
    binFeatPerSession = []
    binLabelPerSession = []
    for ins in range(len(listAllInstances)):
        listPerInstance = listAllInstances[ins]
        binFeat = pd.DataFrame()
        binLabel = pd.DataFrame()
        accDataReceived = False
        for dataSource in range(len(listPerInstance)):
            df = listPerInstance[dataSource]
            df.fillna({'Condition': 0}, inplace=True)
            if 'BVP' in df.columns:
                df = filter_bvp(df, overwrite_bvp=False)
            if 'EDA' in df.columns:
                df = filter_eda(df, overwrite_eda=False)
            if o_is_new_dataset:
                df.fillna({'AGG':0, 'ED':0, 'SIB':0, 'Condition':0}, inplace=True)  # fill NaN with 0

                df['AGG'] = pd.to_numeric(df['AGG'])
                df['ED'] = pd.to_numeric(df['ED'])
                df['SIB'] = pd.to_numeric(df['SIB'])
                df['Condition'] = pd.to_numeric(df['Condition'])

                #if multiclass
                df.loc[df['AGG'] == 1, 'AGG'] = 3
                df.loc[df['ED'] == 1, 'ED'] = 1
                df.loc[df['SIB'] == 1, 'SIB'] = 2
                df['Condition'] = df[['AGG', 'ED', 'SIB']].max(axis=1)
                # Adding the norm of accelerometer data to the data frame.
                # if not accDataReceived:
                if 'ACC_X' in df:
                    acc_data = (df[['ACC_X', 'ACC_Y', 'ACC_Z']]).to_numpy().astype(np.float)
                    df['ACCNorm'] = np.linalg.norm(acc_data, axis=1)
            else:
                df['Condition'][df['Condition'].isin(aggCategory)] = 1  # 1 to Agg state
                df['Condition'][df['Condition'].isin(nonAggCategory)] = 0  # 0 to nonAgg state

                # Adding the norm of accelerometer data to the data frame.
                # if not accDataReceived:
                if 'X' in df:
                    acc_data = (df[['X', 'Y', 'Z']]).to_numpy().astype(np.float)
                    df['ACCNorm'] = np.linalg.norm(acc_data, axis=1)

            if df['Condition'].nunique() > 2:
                print(df['Condition'].nunique())
                assert ('Have unknown labels!')

            # evidence = list(df.columns.values)
            evidence =  [col for col in df.columns if col != 'patientid_session']
            if o_is_new_dataset:
                # Yuna removed patient_id and session from evidence
                for tag in ['Condition', 'Note', 'AGG', 'ED', 'SIB', 'patient_id', 'session']:
                    try:
                        evidence.remove(tag)
                    except:
                        pass # it doesnt have the tag anyways
            else:
                for tag in ['Condition', 'patient_id', 'session']:
                    try:
                        evidence.remove(tag)
                    except:
                        pass
            if len(evidence) > 1:
                accDataReceived = True
            else:
                pass
            for e in range(len(evidence)):
                # count_1 += 1
                binFeat, binLabel = gen_stats_features(df, evidence[e], binFeat, binSize, binLabel)
            
            # Generate PPG features
            if 'BVP' in df.columns:
                df['BVP'] = df['BVP'].astype(float)
                df_hr, df_rmssd = gen_ppg_features(df)
                binFeat, binLabel = gen_stats_features(df_hr, 'HR', binFeat, binSize, binLabel)
                binFeat, binLabel = gen_stats_features(df_rmssd, 'RMSSD', binFeat, binSize, binLabel)
            if 'EDA' in df.columns:
                df_phasic_tonic = gen_eda_features(df)
                binFeat, binLabel = gen_stats_features(df_phasic_tonic, 'PHASIC', binFeat, binSize, binLabel)
                binFeat, binLabel = gen_stats_features(df_phasic_tonic, 'TONIC', binFeat, binSize, binLabel)

        binLabel = binLabel.apply(lambda x: x.astype(float).dropna().max(), axis=1)

        # Add elapsed time to/till aggression features       
        default_tpa = 5000
        if np.sum(binLabel) == 0:
            # no Aggression obeserved!
            binFeat['AGGObserved'] = 0
            binFeat['TimePastAggression'] = default_tpa # trying 5000 to see if it makes any difference
            # binFeat['TimePastAggression'] = 50000000
        else:
            aggObservedFeat = binLabel.copy()
            timePastAggFeat = binLabel.copy()
            aggInd = np.where(binLabel != 0)[0]  # index of aggression Labels
            minAgg = np.min(aggInd)  # get index of first Agg episode

            agg_started = 0
            agg_ended = 1
            agg_initiated = 0
            counter = 0
            for t in range(0, len(binLabel)):
                # if binLabel[t] > 0:
                #     agg_started = True
                # if agg_started and binLabel[t] == 0:
                #     # agg is over
                #     agg_started = False
                #     counter = 0
                #     # aggObservedFeat[t] = 1
                #     agg_happened = True

                # if not agg_happened:
                #     aggObservedFeat[t] = 0
                #     timePastAggFeat[t] = 50000000
                # else:
                #     aggObservedFeat[t] = 1
                #     timePastAggFeat[t] = counter

                # counter += 1
                if onset:
                    if t >= minAgg:
                        aggObservedFeat[t] = 1
                        if t in aggInd:
                            counter = 0
                        else:
                            pass
                        timePastAggFeat[t] = counter
                        counter = counter + 1
                    else:
                        aggObservedFeat[t] = 0
                        timePastAggFeat[t] = default_tpa
                        # timePastAggFeat[t] = 50000000 # trying 5000 to see if it makes any difference
#                 else:
#                     if binLabel[t] > 0:
#                         agg_started = True
#                     if agg_started and binLabel[t] == 0:
#                         # agg is over
#                         agg_started = False
                        
#                         counter = 0
#                         # aggObservedFeat[t] = 1
#                         agg_happened = True
    
#                     if not agg_happened:
#                         aggObservedFeat[t] = 0
#                         timePastAggFeat[t] = default_tpa
#                     else:
#                         aggObservedFeat[t] = 1
#                         timePastAggFeat[t] = counter
                
                #################################################################
                # BRIGGS CHANGES 10/24/24 -- fixes error in offset logic
                #################################################################
                else:
                    if t > 0:
                        if not(binLabel[t]) and binLabel[t-1]:
                            agg_initiated = 1
                            counter = 1
                        else:
                            counter += agg_initiated

                    if not agg_initiated:
                        timePastAggFeat[t] = default_tpa
                        aggObservedFeat[t] = agg_initiated
                    else:
                        timePastAggFeat[t] = counter-1
                        aggObservedFeat[t] = agg_initiated
                #################################################################
                
            # adding new time since last agg features
            binFeat['AGGObserved'] = aggObservedFeat
            binFeat['TimePastAggression'] = timePastAggFeat

            #################################################################
            # BRIGGS CHANGES 10/24/24
            #################################################################
            use_alternative_default_tpa = False
            if use_alternative_default_tpa:
                max_tpa_in_session = binFeat['AGGObserved'].max()
                binFeat['AGGObserved'][binFeat['AGGObserved'] == default_tpa] = max_tpa_in_session

                # mean_tpa_in_session = timePastAggFeat[timePastAggFeat != default_tpa].mean()
                # binFeat['AGGObserved'][binFeat['AGGObserved'] == default_tpa] = mean_tpa_in_session
            #################################################################
                
            
        if not accDataReceived:
            for ax in ['X', 'Y', 'Z']:
                binFeat[ax + 'first'] = 0
                binFeat[ax + 'last'] = 0
                binFeat[ax + 'max'] = 0
                binFeat[ax + 'min'] = 0
                binFeat[ax + 'mean'] = 0
                binFeat[ax + 'median'] = 0
                binFeat[ax + 'nunique'] = 0
                binFeat[ax + 'std'] = 0
                binFeat[ax + 'sum'] = 0
                binFeat[ax + 'var'] = 0
        properOrderOfFeats = binFeat.columns
        
        # Briggs change 11/6 - sets patient session & id as multilevel index
        binFeat['patient_id'] = df['patient_id'].iloc[0]
        binFeat['session'] = df['session'].iloc[0]
        binFeat.set_index(['patient_id', 'session'], append=True, inplace=True)
        binFeat = binFeat.reorder_levels(['patient_id', 'session', binFeat.index.names[0]])
        
        binFeatPerSession.append(binFeat[properOrderOfFeats])    
        binLabelPerSession.append(binLabel)

    outputDict = {'features': binFeatPerSession, 'labels': binLabelPerSession}
    return outputDict


def feature_extraction_csv_dir(dir, bin_size, agg_cat, non_agg_cat, onset, path_style='/', o_is_new_dataset=False):
    """
    This function performs features extraction for all csv files in directory 'dir'. It groups by user's id and sessions.
     Ids are the first 4 digits of folders names containing the csv files. Data from each folder are treated as
     belonging to different sessions. The function returns a dictionary 'data_dict' whose keys() are the user ids. Each
     key points to an other dictionary with keys 'features' and 'labels' and each key points to a list of pandas data
     frames (one for each session),  e.g., pandas_df_features_for_session_0 = data_dict['3458']['fratures'][0]

    Usage:
    data_dict = feature_extraction_csv_dir(dir, bin_size, agg_cat, non_agg_cat, path_style='/', o_is_new_dataset=False)

    Parameters:

    :param dir: main directory of the dataset
    :param bin_size: string containing the sample period (e.g., '15S')
    :param agg_cat: list with possible aggression labels (e.g., agg_cat = ['aggression', 'agg', 'aggression (?)',
                                                                            'property destruction', 'noncompliance'])
    :param non_agg_cat: list with non-aggression labels (e.g., non_agg_cat = ['sleeping', 'mild ED/shoved by peer'])
    :param path_style: '/' for unix style or '\' for MS windows.
    :param o_is_new_dataset: boolean.
    :return: data_dict: dictionary containing 'features' and 'labels' for all sessions and users ids.
    """
    # list all folders in dir
    folders = [x[0] for x in os.walk(dir)]
    # get folders max depth
    max_dir_depth = max([len(folders[i].split(path_style)) for i in range(len(folders))])
    # remove other lesser deep paths from the folder list
    relevant_folders = [folders[i] for i in range(len(folders)) if len(folders[i].split('/')) == max_dir_depth]
    # sort by the last part of the path
    relevant_folders.sort(key=lambda x: x.split(path_style)[-1])

    # get uids
    # uids = get_uid_dict_from_dir_list(relevant_folders, data_path)
    uids = list(set([relevant_folders[i].split(path_style)[-1].split('.')[0]
                     for i in range(len(relevant_folders))]))
    # uid_dict = {i: uids[i] for i in range(len(uids))}

    data_dict = {}
    for uid in uids:
        # get folder list with the same id
        if o_is_new_dataset:
            uid_folder_list = [relevant_folders[i] for i in range(len(relevant_folders)) if
                               relevant_folders[i].split(path_style)[-1].split('.')[0] == str(uid)]
        else:
            # pdb.set_trace()
            print("relevant_folders", relevant_folders)
            uid_folder_list = [relevant_folders[i] for i in range(len(relevant_folders)) if
                      relevant_folders[i].split(path_style)[-1].split('.')[0] == str(uid)]
            print("uid_folder_list",uid_folder_list)
        uid_data_list = []
        # process data for the uid
        for folder_count in range(len(uid_folder_list)):
            # load all csv files in folder
            # print("glob.glob(uid_folder_list[folder_count] + path_style + '*.csv')", uid_folder_list[folder_count] + path_style + '*.csv')
            csv_file_list = sorted(glob.glob(uid_folder_list[folder_count] + path_style + '*.csv'), reverse=True,
                                   key=os.path.getsize)
            # Yuna added - remove combined files to avoid overwriting features
            csv_file_list = [file for file in csv_file_list if 'combined' not in file]
            if len(csv_file_list) == 0:#the data folder that doesn't have any sessions
                continue
            # sorting to have an ACC followed by an EDA file for each session
            # csv_file_list.sort(key=lambda x: x.split('_')[3])
            csv_file_list.sort(key=lambda x: x.split(path_style)[-1].split('_')[1])

            # count the number of different files for sessions
            # tt = set([csv_file_list[i].split('.').split('_')[-2] for i in range(len(csv_file_list))])
            # pdb.set_trace()
            if o_is_new_dataset:
                n_files_per_session = len(set([csv_file_list[i].split('_')[-2] for i in range(len(csv_file_list))]))
            else:
                n_files_per_session = len(set([i.split('_')[-1].split('.')[0] for i in csv_file_list]))
            print("csv_file_list", csv_file_list)
            # pdb.set_trace()
            n_sessions = int(len(csv_file_list)/n_files_per_session)

            # process each session in the folder
            for s in range(n_sessions):
                # create a session file list
                session_file_list = csv_file_list[0 + s * n_files_per_session:n_files_per_session + s * n_files_per_session]
                print(session_file_list)

                list_per_instance = []
                print(csv_file_list)

                # for file in csv_file_list:
                for file in session_file_list:
                    print(file)
                    # loading each csv file
                    df = pd.read_csv(file, index_col=None, header=0, dtype=object)
                    # using Timestamp as index
                    # return df
                    try:
                        df_index_time = df.set_index('Timestamp')
                    except KeyError as ke:
                        raise(ke)
                    # the command bellow was done so that pandas could recognize dfIndexTime as a time index
                    if o_is_new_dataset:
                        df_index_time = df_index_time.set_index(
                            pd.to_datetime(pd.to_numeric(df_index_time.index), origin='unix', unit='ms'))
                    else:
                        # pdb.set_trace()
                        df_index_time = df_index_time.set_index(pd.to_datetime(df_index_time.index))
                    # appending all different data in one userList in which each element
                    # is a different instance (e.g., EDA, IBI, etc)
                    
                    # Briggs added 11/6 - input patient_id and session into df
                    match = re.search(r'/(\d+\.\d+)_([\d]{2})_', file)
                    patient_id = match.group(1)
                    session = match.group(2)
                    df_index_time['patient_id'] = patient_id
                    df_index_time['session'] = session

                    print(df_index_time)

                    list_per_instance.append(df_index_time)

                uid_data_list.append(list_per_instance)
        user_dict = {"dataAll": uid_data_list}
        # print(user_dict)
        data_dict[uid] = feat_generator(user_dict, bin_size, agg_cat, non_agg_cat, onset = onset, o_is_new_dataset=o_is_new_dataset)
    return data_dict


# def feature_extraction(dir, bin_size, agg_cat, non_agg_cat, feat_code, path_style='/',
#                        o_is_new_dataset=False, o_run_from_scratch=False, o_multiclass=False):

def feature_extraction(dir, bin_size, agg_cat, non_agg_cat, feat_code, onset, o_is_new_dataset=False, o_run_from_scratch=False, o_multiclass=False, path_style='/'):
    """
    Main feature extraction function. This function reads files from the dataset directory 'dir' and returns two
    dictionaries: a data_dict, with dataframes of features and labels for all users and all sessions, and a uid_dict
    with the ids of each user. The function also saves a binary version of the outputs so its save time when re-running
    a simulation. This binary datafile is stored in 'dir' and name is bin_feat_<bin_size>_<feat_code>.b. If this file is
    present then this function will only load this file and return the appropriate dictionaries unless the boolean
    variable o_run_from_scratch is True.
    :param dir: main directory of the dataset
    :param bin_size: string containing the sample period (e.g., '15S')
    :param agg_cat: list with possible aggression labels (e.g., agg_cat = ['aggression', 'agg', 'aggression (?)',
                                                                            'property destruction', 'noncompliance'])

    :param non_agg_cat: list with non-aggression labels (e.g., non_agg_cat = ['sleeping', 'mild ED/shoved by peer'])
    #
    # :param feat_code: integer selecting the features according to select_feat_from_feat_code() function.
    #                   Current options are :  -1, for using only TimePastAggression
    #                                          0, for ['AGGObserved', 'TimePastAggression']
    #                                          1, for ACC only
    #                                          2, for EDA only
    #                                          3, for BVP only
    #                                          4, for IBI only (if applicable: the new dataset does not have IBI files)
    #                                          5, for BVP + EDA
    #                                          6, for all signals (ACC+EDA+BVP +IBI(if available))
    #                                          7, for all signals + ['AGGObserved', 'TimePastAggression']
    #                                          8, for 'ACCNormmean'
    #                                          9, for BVP + EDA + 'AGGObserved', 'TimePastAggression'
    #                                          10, for ACC + EDA + 'AGGObserved', 'TimePastAggression'
    #                                          11, for ACC + BVP + 'AGGObserved', 'TimePastAggression'
    #                                          12, for 'AGGObserved', 'TimePastAggression'
    #                                          13, for 'ACC', 'AGGObserved', 'TimePastAggression'
    #                                          14, for all signals (ACC+EDA+BVP+HR+RMSSD+TONIC+PHASIC)
    #                                          15, for all signals (ACC+EDA+BVP+HR+RMSSD+TONIC+PHASIC) + ['AGGObserved', 'TimePastAggression']
    :param path_style: '/' for unix style or '\' for MS windows.
    :param o_is_new_dataset:  boolean.
    :param o_run_from_scratch:  boolean.
    :param o_multiclass: boolean.
    :return: data_dict, uid_dict
    """
    # this is the main function
    # if o_multiclass:
    #     feature_file_name = dir + path_style + 'bin_feat_' + bin_size + '_' + str(feat_code) + '_mc.b'
    # else:
    #     feature_file_name = dir + path_style + 'bin_feat_' + bin_size + '_' + str(feat_code) + '.b'
    
    # path_style='/'
    if o_multiclass:
        feature_file_name = dir + path_style + 'bin_feat_' + str(bin_size) + "_onset" + str(onset) + '_mc.b'
    else:
        feature_file_name = dir + path_style + 'bin_feat_' + str(bin_size) + "_onset" + str(onset) + '.b'

    # print("!!!!!!!!!!HHHHEEEEERRRREEEEEE11111!!!!!!!!!!!!!")
    # print(feature_file_name)

    # uid_dict = get_uid_dict_from_dir(dir, path_style, id_ndigits=id_ndigits)
    if not os.path.isfile(feature_file_name) or o_run_from_scratch:
        data_dict = feature_extraction_csv_dir(dir, bin_size, agg_cat, non_agg_cat, onset, path_style='/',
                                               o_is_new_dataset=o_is_new_dataset)
        pickle_out = open(feature_file_name, 'wb')
        pickle.dump(data_dict, pickle_out)
    else:
        print('loading data...')
        pickle_in = open(feature_file_name, 'rb')
        data_dict = pickle.load(pickle_in)
    uids = list(data_dict.keys())
    uid_dict = {i: uids[i] for i in range(len(uids))}
    return data_dict, uid_dict
    #1) get dframedict
    #2) for each user id:
        #2.1) extract create instance vectors


def gen_classifier_instances_from_session_data_frame(feat_data_frame_per_session, label_data_frame_per_session, num_observation_frames, 
                                                     num_prediction_frames, selected_feat,
                                                     agg_intensity_clf_per_sbj=None, o_multiclass=False, overlap=0):
    # using only the selected features
    print("selected_feat", selected_feat)
    feat_data_frame_per_session_sfeat = feat_data_frame_per_session[selected_feat]
    n_different_features = len(selected_feat)

    # removing NaN from extracted features
    feat_data_frame_per_session_sfeat.fillna(0, inplace=True)

    total_num_of_frames = len(feat_data_frame_per_session_sfeat)
    n_instances = total_num_of_frames - num_prediction_frames - num_observation_frames + 1
    # idxs = feat_data_frame_per_session.index

    # for loop over all possible frames
        # instance [i] = concatenation of num_observation_frames
        # label [i] = 0 or 1 by looking at the future num_prediction_frames
    print('n_instances ', n_instances)
    if n_instances < 1:
        print("Too few samples, session is being droped!")
        return None, None, None, None
    # instances_array = np.zeros((n_instances, len(selected_feat)*num_observation_frames))
    # labels_array = np.zeros((n_instances, 1))

    # Clustering ACCNormmean to augment the labels.
    if agg_intensity_clf_per_sbj is not None:
        acc_feats = ['ACCNormmean']
        try:
            X = feat_data_frame_per_session[acc_feats]
        except:
            return None, None, None, None
        X = np.array(X)
        # X = np.linalg.norm(X, axis=1)

        y = label_data_frame_per_session.to_numpy().astype(int)
        # agg_idx = y > 0
        agg_idx = np.where(y > 0)[0]
        if agg_idx.shape[0] != 0:
            print('here')
            X = X[agg_idx]
            km_pred = agg_intensity_clf_per_sbj.predict(X.reshape(-1, 1))
            for cc in range(agg_intensity_clf_per_sbj.n_clusters):
                y[agg_idx[km_pred == cc]] = cc + 1
            label_data_frame_per_session = pd.DataFrame(y)

    instances_list = []
    labels_list = []
    instance_count = 0
    indices = []
    raw_label_list = []

    for i in range(0, n_instances):
        a = feat_data_frame_per_session_sfeat.iloc[i: i + num_observation_frames]
        t = 'TimePastAggression' in a.columns
        #or 'AGGObserved' in a.columns
        # if t:
        #     if a.TimePastAggression.iloc[-1] == 0:
        #         continue
        temp_np_array = np.array(feat_data_frame_per_session_sfeat.iloc[instance_count: instance_count + num_observation_frames])
        instances_list.append(np.reshape(temp_np_array, [1, -1], order='C'))
        
        # Briggs NOTE each index is (patient_id, session, timestamp) for the MOST RECENT 15sec (i.e. t-15s) bin in observation frame
        indices.append(feat_data_frame_per_session_sfeat.iloc[instance_count + num_observation_frames].name)
        # indices.append(feat_data_frame_per_session_sfeat.iloc[instance_count].name)# use for first 15sec bin (ex. t-180s)

        # instances_list.append(temp_np_array)
        
        # Briggs NOTE confirms that ANY aggression within the prediction frame gets counted as aggression
        labels_list.append(label_data_frame_per_session.iloc[
                          instance_count + num_observation_frames: instance_count + num_observation_frames + num_prediction_frames].max())

        raw_label_list.append((label_data_frame_per_session.iloc[instance_count + num_observation_frames + num_prediction_frames - 1]))

        instance_count += 1

        # temp_np_array = np.array(feat_data_frame_per_session_sfeat.iloc[i: i + num_observation_frames])
        # instances_array[i] = np.reshape(temp_np_array, [1, -1], order='C')
        #
        # labels_array[i] = label_data_frame_per_session.iloc[
        #                   i + num_observation_frames: i + num_observation_frames + num_prediction_frames].max()
        # if not o_multiclass:
        #     if agg_intensity_clf_per_sbj is None:
        #         if labels_array[i] > 0:
        #             labels_array[i] = 1
        #         else:
        #             labels_array[i] = -1
        #     else:
        #         if labels_array[i] == 0:
        #             labels_array[i] = -1
    #Ahmet:Add -1 to remove bug
    if not o_multiclass:
        if o_multiclass is None:
            if labels_list[instance_count-1] > 0:
                labels_list[instance_count-1] = 1
            else:
                labels_list[instance_count-1] = -1
        # else:
            # pdb.set_trace()
            # if labels_list[instance_count-1] == 0:
            #     labels_list[instance_count-1] = 0#Ahmet:was -1
    if not instances_list:
        return None, None, None
    instances_array = np.vstack(instances_list)
    labels_array = np.vstack(labels_list)

    # add std of each feature
    # each feature repeats every len(selected_feat)
    # feat_idx = [0  n_different_features n_different_features*2,...]
    n_instances = len(labels_array)
    feat_std_array = np.zeros([n_instances, n_different_features])
    for feat_count in range(n_different_features):
        idx = [feat_count + n_different_features*cc for cc in range(0, num_observation_frames)]
        feat_std_array[:, feat_count] = np.std(instances_array[:, idx], axis=1)
    instances_array = np.concatenate((instances_array, feat_std_array), axis=1)
    
    # setting feature names
    non_summary_stat_features = {'AGGObserved', 'TimePastAggression'}
    feat_col_names_std = [f'{selected_feat[i]}_{num_observation_frames*15}s_window_std_dev' for i in range(len(selected_feat))]
    bin_size=15
    col_names_all_frames = [f"{col}_t-{i}s" for i in range(num_observation_frames*bin_size,0,-bin_size) for col in selected_feat]
    feat_col_names = col_names_all_frames+feat_col_names_std
    
    # setting a multi-level index for patient_id, session, and timestamp
    instance_df = pd.DataFrame(instances_array, columns=feat_col_names)
    index = pd.MultiIndex.from_tuples(indices, names=['patient_id', 'session', 'Timestamp'])
    instance_df.index = index

    # input raw label into df
    instance_df[f'rawLabel_t+{num_prediction_frames*15}s'] = raw_label_list
    instance_df[f'processedLabel_{num_prediction_frames*15/60:.0f}min_future_window'] = labels_list
    
    return instances_array, labels_array, feat_col_names, instance_df


def gen_instances_from_raw_feat_dictionary(feat_dict, num_observation_frames, num_prediction_frames, feat_code, onset,
                                           agg_intensity_clf=None, o_is_new_dataset=False, o_multiclass=False,
                                           o_return_list_of_sessions=False, outdir="." ,o_run_from_scratch=False,
                                           bin_size='15S'):
    """
    This function creates arrays of features and labels in ndarrays
    :param feat_dict:
    :param num_observation_frames:
    :param num_prediction_frames:
    :param feat_code:
    :param agg_intensity_clf:
    :param o_is_new_dataset:
    :param o_multiclass: Boolean (default: False)
    :param o_return_list_of_sessions: Boolean (default: False) This will split the data into a list of sessions
    :return: if o_return_list_of_sessions is False returns data and label dictionaries X[id], y[id] with data of all
    sessions concatenated in a single np.array for each id.
                        X[id] is a 2D ndarray of features (all sessions concatenated)
                        y[id] is a 1D array of labels (all sessions concatenated)
    If o_return_list_of_sessions is True, returns data and label dictionaries X[id], y[id] containing lists of numpy
    arrays. Each list element is a np.array of data or labels for each session.
                        X[id][i] is a 2D matrix of features for session i.
                        y[id][i] is a 1D array of labels corresponding to the i-th session.
    """
    # select features from feat_code
    selected_feat = select_feat_from_feat_code(feat_code, 
                                               o_is_new_dataset=o_is_new_dataset)

    print(selected_feat)
    # print(len(selected_feat)) 
    dict_of_instances_arrays = {}
    dict_of_labels_arrays = {}
    dict_of_superposition_lists = {}

    # Briggs added 11/6
    dict_of_session_dfs ={}

    id_blacklist = []

    filename = outdir + "/dataInst_fc" + str(feat_code) + "_tp" + str(num_observation_frames) + "_tp" \
               + str(num_prediction_frames) + "_mc" + str(o_multiclass) + "_rs" + str(o_return_list_of_sessions) \
               + '_bs' + str(bin_size) + "_ons" + str(onset) + ".bin"

    # filename = outdir + "/dataInst_fc" + "_tp" + str(num_observation_frames) + "_tp" \
    #            + str(num_prediction_frames) + "_mc" + str(o_multiclass) + "_rs" + str(o_return_list_of_sessions) \
    #            + '_bs' + str(bin_size) + ".bin"
    # o_run_from_scratch = 
    o_run_from_scratch = True
    if (not o_run_from_scratch) and os.path.isfile(filename):
        # load file
        print('loading data instance data...')
        pickle_in = open(filename, 'rb')
        datalist = pickle.load(pickle_in)
        dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, dict_of_superposition_lists, dict_of_session_dfs = datalist
    else:
        # loops over each subject
        for subjID in feat_dict:
            dict_of_superposition_lists[subjID] = []
            try:
                if agg_intensity_clf is not None:
                    try:
                        agg_intensity_clf_per_sbj = agg_intensity_clf[subjID]
                        if agg_intensity_clf_per_sbj is None:
                            print(f"Subject {subjID} has no aggression classifier.")
                            continue
                    except:
                        print('has no aggression classifier for ID ' + str(subjID) + '.')
                        agg_intensity_clf_per_sbj = None
                else:
                    agg_intensity_clf_per_sbj = None

                print('subjID', subjID)
                
                if o_return_list_of_sessions:
                    instances_array_per_subject_list = []
                    labels_array_per_subject_list = []
                    dfs_array_per_subject_list = []
                else:
                    instances_array_per_subject = np.array([]).reshape(0, num_observation_frames * len(selected_feat) +
                                                                       len(selected_feat))
                    labels_array_per_subject = np.array([]).reshape(0, 1)

                feat_dic_per_subj = feat_dict[subjID]

                # print(len(feat_dic_per_subj['features']), len(feat_dic_per_subj['labels']))

                # getting list of features and labels from each the feature dictionary from subject.
                feat_list_per_subj = feat_dic_per_subj['features']
                label_list_per_subj = feat_dic_per_subj['labels']

                sup_list = []
                # loop over sessions for a given subject
                for i in range(len(label_list_per_subj)):
                    print("session " + str(i))
                    # getting pandas data frame for each session within a subject
                    feat_data_frame_per_session = feat_list_per_subj[i]
                    label_data_frame_per_session = label_list_per_subj[i]

                    # print(feat_data_frame_per_session)
                    session_instances_array, session_labels_array,feat_col_names, instance_df = gen_classifier_instances_from_session_data_frame(
                        feat_data_frame_per_session, label_data_frame_per_session, num_observation_frames,
                        num_prediction_frames, selected_feat, agg_intensity_clf_per_sbj, o_multiclass=o_multiclass)

                    if session_instances_array is None:
                        continue

                    # append list of superposision_index for current
                    sup_list += gen_superposition_index_list(len(session_instances_array), num_observation_frames)

                    if o_return_list_of_sessions:
                        instances_array_per_subject_list.append(session_instances_array)
                        labels_array_per_subject_list.append(session_labels_array)
                        dict_of_superposition_lists[subjID].append(sup_list)
                        sup_list = []
                        
                        dfs_array_per_subject_list.append(instance_df)
                    
                    else:
                        instances_array_per_subject = np.concatenate((instances_array_per_subject, session_instances_array), axis=0)
                        labels_array_per_subject = np.concatenate((labels_array_per_subject, session_labels_array), axis=0)

                if not o_return_list_of_sessions:
                    dict_of_superposition_lists[subjID] = sup_list

                if o_return_list_of_sessions:
                    dict_of_instances_arrays[subjID] = instances_array_per_subject_list
                    dict_of_labels_arrays[subjID] = labels_array_per_subject_list
                    dict_of_session_dfs[subjID] = dfs_array_per_subject_list
                else:
                    dict_of_instances_arrays[subjID] = instances_array_per_subject
                    dict_of_labels_arrays[subjID] = labels_array_per_subject


            # NOTE Zulqurnain suggested changes
            except AssertionError as error:
            # except Exception as error:
                print(error)
                id_blacklist.append(subjID)
                print('Not possible to construct data for sbj ' + subjID)

        datalist = [dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, dict_of_superposition_lists, dict_of_session_dfs]
        pickle_out = open(filename, 'wb')
        pickle.dump(datalist, pickle_out)

    # NOTE returning selected_feat & feat_col_names may causes errors
    return dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, dict_of_superposition_lists, selected_feat, feat_col_names, dict_of_session_dfs


def gen_superposition_index_list(num_of_instances, num_observation_frames):
    sup_list = []
    for i in range(num_of_instances):
        # i - num_observation_frames + 1 : i + num_observation_frames - 1
        # sup_list.append(list(range(max(0, i - num_observation_frames + 1),
        #                          min(i + num_observation_frames, num_of_instances))))
        sup_list.append([i - max(0, i - num_observation_frames + 1),
                         min(i + num_observation_frames - 1, num_of_instances) - i])
    return sup_list


def kmeans_accnorm(data_dict, 
                   o_plot_histograms=False, 
                   n_clusters=3,
                   return_data=False):

    # get only norm of accelerometer data
    feat_code = 8

    feats = select_feat_from_feat_code(feat_code)

    # print('loading data...')
    # pickleIn = open(feature_file_name, 'rb')
    #
    # # dictionary of data frames
    # data_dict = pickle.load(pickleIn)

    count = 1
    kmeans_dict = {}
    for key in data_dict.keys():

        sbj_data_dict = data_dict[key]
        sbj_feat_list = sbj_data_dict['features']
        sbj_label_list = sbj_data_dict['labels']

        sbj_feat_array = []
        sbj_label_array = []
        print("Processing Key ", key)
        try:
            for i in range(len(sbj_feat_list)):
                # tmp_array = np.linalg.norm(np.array(sbj_feat_list[i][['Xmean', 'Ymean', 'Zmean']]), axis=1)
                if 'ACCNormmean' in sbj_feat_list[i]:
                    tmp_array = np.array(sbj_feat_list[i]['ACCNormmean'])
                    print("loaded ACCNORMMEAN for ", key)
                    sbj_feat_array = np.concatenate((sbj_feat_array, tmp_array), axis=0)
                    sbj_label_array = np.concatenate((sbj_label_array, sbj_label_list[i].astype(float)), axis=0)
            if len(sbj_label_array) > 0 :
                X = sbj_feat_array[sbj_label_array >= 1].reshape(-1, 1)
                X = np.nan_to_num(X)
                kmeans = None
                # check if there is enough data...
                if len(X) > n_clusters*10:
                    # kmeans = KMeans(init= np.array([min(X), max(X)]), n_clusters=2, random_state=0).fit(X)
                    if n_clusters == 3:
                        # kmeans = KMeans(init=np.array([min(X), (min(X) + max(X)) / 2, max(X)]), n_clusters=n_clusters, random_state=0).fit(X)
                        # print(X)
                        
                        kmeans = KMeans(n_clusters=n_clusters).fit(X)
                        means = kmeans.cluster_centers_.copy()
                        klabels = kmeans.labels_.copy()
                        for i in range(n_clusters):
                            idx = np.argmax(means)
                            kmeans.labels_[klabels == idx] = n_clusters - i -1
                            means[idx] = -10000
    
                        kmeans.cluster_centers_ = np.array(sorted(kmeans.cluster_centers_))
    
                    elif n_clusters == 2:
                        # kmeans = KMeans(init=np.array([min(X), max(X)]), n_clusters=n_clusters,
                        #                 random_state=0).fit(X)
                        kmeans = KMeans(n_clusters=n_clusters,
                                        random_state=0).fit(X)
    
                        means = kmeans.cluster_centers_
                        if means[0] > means[1]:
                            print('KMeans Centroids not in crescent order. Inverting kmmean labels!')
                            kmeans.labels_ = abs(kmeans.labels_ -1)
                            kmeans.cluster_centers_ = np.array(sorted(kmeans.cluster_centers_))
    
                    else:
                        kmeans = KMeans(n_clusters=n_clusters,
                                        random_state=0).fit(X)
    
                kmeans_dict[key] = kmeans
    
                if o_plot_histograms:
                    km_pred = kmeans.predict(X)
                    # plt.figure()
                    # plt.hist(X[km_pred == 0], bins=20)
                    # plt.hist(X[km_pred == 1], bins=20)
                    # plt.hist(X[km_pred == 2], bins=20)
                    # if count == 8:
                    #     print(kmeans)
                    #     print(X.T)
                    #     print(km_pred)
                    #     break
                    # count += 1
                    plt.figure()
                    plt.hist(X[km_pred == 0])
                    plt.hist(X[km_pred == 1])
                    plt.hist(X[km_pred == 2])
                    # if count == 8:
                    #     print(kmeans)
                    #     print(X.T)
                    #     print(km_pred)
                    #     break
                    # count += 1
                    plt.show()
    
                    print(kmeans.cluster_centers_)
        except AssertionError as error:
            print('Except!')
            print(error)
            print('Could not process ' + str(key))
            print(X.shape[0])
            print(sbj_label_array)

    if return_data:
        return kmeans_dict, data_dict
    return kmeans_dict


def pandas_dataframe_from_csv_dir(dir, path_style='/', o_is_new_dataset=False):
    """
    This function performs features extraction for all csv files in directory 'dir'. It groups by user's id and sessions.
     Ids are the first 4 digits of folders names containing the csv files. Data from each folder are treated as
     belonging to different sessions. The function returns a dictionary 'data_dict' whose keys() are the user ids. Each
     key points to an other dictionary with keys 'features' and 'labels' and each key points to a list of pandas data
     frames (one for each session),  e.g., pandas_df_features_for_session_0 = data_dict['3458']['fratures'][0]

    Usage:
    data_dict = feature_extraction_csv_dir(dir, bin_size, agg_cat, non_agg_cat, path_style='/', o_is_new_dataset=False)

    Parameters:

    :param dir: main directory of the dataset
    :param path_style: '/' for unix style or '\' for MS windows.
    :param o_is_new_dataset: boolean.
    :return: data_dict: dictionary indexed by users ids containing list of pandas data frames (one per session).
    """
    # list all folders in dir
    folders = [x[0] for x in os.walk(dir)]
    # get folders max depth
    max_dir_depth = max([len(folders[i].split(path_style)) for i in range(len(folders))])
    # remove other lesser deep paths from the folder list
    relevant_folders = [folders[i] for i in range(len(folders)) if len(folders[i].split('/')) == max_dir_depth]
    # sort by the last part of the path
    relevant_folders.sort(key=lambda x: x.split(path_style)[-1])

    # get uids
    # uids = get_uid_dict_from_dir_list(relevant_folders, data_path)
    uids = list(set([relevant_folders[i].split(path_style)[-1].split('.')[0]
                     for i in range(len(relevant_folders))]))
    # uid_dict = {i: uids[i] for i in range(len(uids))}

    data_dict = {}
    for uid in uids:
        # get folder list with the same id
        uid_folder_list = [relevant_folders[i] for i in range(len(relevant_folders)) if
                           relevant_folders[i].split(path_style)[-1].split('.')[0] == str(uid)]

        uid_data_list = []
        # process data for the uid
        for folder_count in range(len(uid_folder_list)):
            # load all csv files in folder
            csv_file_list = sorted(glob.glob(uid_folder_list[folder_count] + path_style + '*.csv'), reverse=True,
                                   key=os.path.getsize)
            # sorting to have an ACC followed by an EDA file for each session
            # csv_file_list.sort(key=lambda x: x.split('_')[3])
            csv_file_list.sort(key=lambda x: x.split(path_style)[-1].split('_')[1])

            # count the number of different files for sessions
            # tt = set([csv_file_list[i].split('.').split('_')[-2] for i in range(len(csv_file_list))])
            if o_is_new_dataset:
                n_files_per_session = len(set([csv_file_list[i].split('_')[-2] for i in range(len(csv_file_list))]))
            else:
                n_files_per_session = len(set([csv_file_list[i].split('_')[-1].split('.')[0] for i in range(len(csv_file_list))]))
            n_sessions = int(len(csv_file_list)/n_files_per_session)

            # process each session in the folder
            for s in range(n_sessions):
                # create a session file list
                session_file_list = csv_file_list[0 + s * n_files_per_session:n_files_per_session + s * n_files_per_session]
                print(session_file_list)

                list_per_instance = []
                print(csv_file_list)

                # for file in csv_file_list:
                for file in session_file_list:
                    print(file)
                    # loading each csv file
                    df = pd.read_csv(file, index_col=None, header=0, dtype=object)
                    # using Timestamp as index
                    # return df
                    df_index_time = df.set_index('Timestamp')
                    # the command bellow was done so that pandas could recognize dfIndexTime as a time index
                    if o_is_new_dataset:
                        df_index_time = df_index_time.set_index(
                            pd.to_datetime(pd.to_numeric(df_index_time.index), origin='unix', unit='ms'))
                    else:
                        df_index_time = df_index_time.set_index(pd.to_datetime(df_index_time.index))
                    # appending all different data in one userList in which each element
                    # is a different instance (e.g., EDA, IBI, etc)
                    print(df_index_time)
                    list_per_instance.append(df_index_time)

                uid_data_list.append(list_per_instance)
        data_dict[uid] = uid_data_list
        # print(user_dict)
        # data_dict[uid] = user_dict
    return data_dict


if __name__ == '__main__':
    # data_path = '/home/tales/DataBases/new_CBS_data'
    # data_path = '/home/tales/DataBases/smallASDData/Data'
    # data_path = '/home/tales/DataBases/new_data_small'
    o_is_new_dataset = True
    
    if o_is_new_dataset :
        if platform.system() == "Windows":
            data_path = 'Z:/1123/aggression_after_review/CBS_DATA_ASD_ONLY'
            path_style = '\\'
        else:
            data_path = '/scratch/demirkaya.a/1123/aggression_after_review/CBS_DATA_ASD_ONLY'
            path_style = '/'
        # data_path = "Z:/1023/aggression_after_review/CBS_DATA_ASD_ONLY"
    else:
        data_path = '/scratch/demirkaya.a/agression_2022_v2/github_repo/dataset/'
        path_style = '\\'
    data_path = '/home/tales/DataBases/new_data_t1'

    # path_style = '/'
    subjectIDCoding = 4  # Number of digits in subject ID coding

    agg_cat = ['aggression', 'agg', 'aggression (?)', 'property destruction', 'noncompliance',
                   'SIB', 'ED', 'kicked table', 'aggression/SIB', 'elopement/drop on the floor',
                   'aggression/SIB/property destruction', 'dropped on floor', 'sib', 'elopement',
                   'AGG', 'Agg', 'Agg/SIB']
    non_agg_cat = ['sleeping', 'mild ED/shoved by peer']

    bin_size = '15S'  # Frame size in seconds

    feat_code = 6

    # num_observation_frames = 12
    # num_prediction_frames = 4
    num_observation_frames = 12
    num_prediction_frames = 4

    ll = select_feat_from_feat_code(feat_code, o_is_new_dataset=True)

    data_dict, uid_dict = feature_extraction(data_path, bin_size, agg_cat, non_agg_cat, feat_code, onset, 
                                             path_style=path_style, o_is_new_dataset=o_is_new_dataset,
                                             o_run_from_scratch=False)

    agg_intensity_clf = kmeans_accnorm(data_dict, o_plot_histograms=True, n_clusters=3, return_data=False)

    dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, selected_feat, feat_col_names, dict_of_session_dfs = \
        gen_instances_from_raw_feat_dictionary(data_dict, num_observation_frames, num_prediction_frames, feat_code, onset,
                                               agg_intensity_clf=agg_intensity_clf, o_is_new_dataset=o_is_new_dataset)

    # data_dict, uid_dict = feature_extraction_csv_dir(data_path, bin_size, agg_cat, non_agg_cat,
    #                                                  path_style='/', o_is_new_dataset=o_is_new_dataset)
