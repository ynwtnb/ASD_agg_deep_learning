"""
This module is for extracting data from the raw data files into a format that can be loaded into the models.
"""

import os
import glob
import pandas as pd
import pickle
import re
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser

from physioview.pipeline import PPG, EDA, SQA

import physio_processing as pp


def data_preprocess(data_path, num_observation_frames=12, 
                    num_prediction_frames=4, o_run_from_scratch=False,
                    o_return_list_of_sessions=False,
                    bin_size=15, o_multiclass=True):
    """
    This function preprocesses the data by binning the data and generating instances from the binned data.

    Parameters
    ----------
    data_path : str
        The path to the data directory.
    num_observation_frames : int, optional
        The number of observation frames to include in the data.
        The observation window will be num_observation_frames * bin_size.
    num_prediction_frames : int, optional
        The number of prediction frames to include in the data.
        The prediction window will be num_prediction_frames * bin_size.
    o_run_from_scratch : bool, optional
        Whether to run the data extraction from scratch.
        If False, the data will be loaded from the saved binary file if it exists.
    o_return_list_of_sessions : bool, optional
        Whether to return the data as a list of sessions.
        If True, the data will be returned as a list of sessions.
        If False, the data will be returned as a single numpy array.
    bin_size : int, optional
        The size of the bins to use for the data.
    o_multiclass : bool, optional
        Whether to include the multiclass labels in the data.
        If True, the data will be returned as a multiclass model.
        If False, the data will be returned as a binary model.
    
    Returns
    -------
    dict_of_instances_arrays : dict
        A dictionary of instances arrays.
        The keys are the user IDs. The values are numpy arrays of instances.
    dict_of_labels_arrays : dict
        A dictionary of labels arrays.
        The keys are the user IDs. The values are numpy arrays of labels for each instance.
    id_blacklist : list
        A list of blacklisted IDs. The IDs are the user IDs that were not able to be processed.
    uid_dict : dict
        A dictionary of user IDs. 
    dict_of_superposition_lists : dict
        A dictionary of superposition lists.
        The keys are the user IDs. The values are lists of superposition indices.
        The superposition indices are the indices of the instances that are superimposed on each other.
    feat_col_names : list
        A list of feature column names.
    dict_of_session_dfs : dict
        A dictionary of session dataframes.
    """

    # Load data
    agg_cat = ['AGG','SIB', 'ED']
    # First create binned data from raw data
    # Binned data is a dictionary of dataframes for each user and session
    # The labels are the labels for whether the aggression is observed in each bin
    data_dict, uid_dict = data_extraction(data_path, bin_size, agg_cat,
                                            o_run_from_scratch=o_run_from_scratch,
                                            o_multiclass=o_multiclass)
    # Then generate instances from binned data
    # Instance is a sequence of binned data that is used for training/testing
    # The instance labels are the labels for whether the aggression is observed in the future prediction window
    dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, dict_of_superposition_lists, feat_col_names, dict_of_session_dfs = \
        gen_instances_from_raw_feat_dictionary(data_dict, num_observation_frames, num_prediction_frames,
                                            o_multiclass=o_multiclass,
                                            o_return_list_of_sessions=o_return_list_of_sessions,
                                            outdir=data_path, o_run_from_scratch=o_run_from_scratch, bin_size=bin_size)

    # remove blacklisted IDs from IDsdict
    if len(id_blacklist) != 0:
        for v in id_blacklist:
            for k in uid_dict.keys():
                if uid_dict[k] == v:
                    blacklisted_key = k
            del uid_dict[blacklisted_key]

    return dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, uid_dict, dict_of_superposition_lists, feat_col_names, dict_of_session_dfs

# ============= Modules for binning data =============

def data_extraction(dir, bin_size, agg_cat, o_run_from_scratch=False, o_multiclass=False, path_style='/'):
    """
    Main feature extraction function. This function reads files from the dataset directory 'dir' and returns two
    dictionaries: a data_dict, with dataframes of features and labels for all users and all sessions, and a uid_dict
    with the ids of each user. The function also saves a binary version of the outputs so its save time when re-running
    a simulation. This binary datafile is stored in 'dir' and name is bin_feat_<bin_size>.b. If this file is
    present then this function will only load this file and return the appropriate dictionaries unless the boolean
    variable o_run_from_scratch is True.
    :param dir: main directory of the dataset
    :param bin_size: int containing the sample period in seconds (e.g., 15)
    :param agg_cat: list with possible aggression labels (e.g., agg_cat = ['AGG', 'SIB', 'ED'])
    :param path_style: '/' for unix style or '\' for MS windows.
    :param o_multiclass: boolean.
    :return: data_dict, uid_dict
    """
    # this is the main function
    if o_multiclass:
        feature_file_name = dir + path_style + 'bin_feat_' + str(bin_size) + 'S_mc.b'
    else:
        feature_file_name = dir + path_style + 'bin_feat_' + str(bin_size) + 'S.b'

    if not os.path.isfile(feature_file_name) or o_run_from_scratch:
        data_dict = data_extraction_csv_dir(dir, bin_size, agg_cat, path_style='/')
        pickle_out = open(feature_file_name, 'wb')
        pickle.dump(data_dict, pickle_out)
    else:
        print('loading data...')
        pickle_in = open(feature_file_name, 'rb')
        data_dict = pickle.load(pickle_in)
    uids = list(data_dict.keys())
    uid_dict = {i: uids[i] for i in range(len(uids))}
    return data_dict, uid_dict

def data_extraction_csv_dir(dir, bin_size, agg_cat, path_style='/'):
    """
    This function performs data extraction for all csv files in directory 'dir'. It groups by user's id and sessions.
    Ids are the first 4 digits of folders names containing the csv files. Data from each folder are treated as
    belonging to different sessions. The function returns a dictionary 'data_dict' whose keys() are the user ids. Each
    key points to an other dictionary with keys 'features' and 'labels' and each key points to a list of pandas data
    frames (one for each session),  e.g., pandas_df_features_for_session_0 = data_dict['3458']['fratures'][0]

    Each pandas dataframe contains the data for a single session. Each row is a single bin of bin_size, and each column is a single feature.
    Each cell contains a 1D array of shape (samples_per_bin,) of the feature value for that bin.

    Usage:
    data_dict = data_extraction_csv_dir(dir, bin_size, agg_cat, path_style='/')

    Parameters:
    :param dir: main directory of the dataset
    :param bin_size: int containing the sample period in seconds (e.g., 15)
    :param agg_cat: list with possible aggression labels (e.g., agg_cat = ['AGG', 'SIB', 'ED'])
    :param path_style: '/' for unix style or '\' for MS windows.
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
    uids = list(set([relevant_folders[i].split(path_style)[-1].split('.')[0]
                    for i in range(len(relevant_folders))]))
    # TODO: Delete this after testing
    uids = [uids[0]]

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
            # Remove 'combined' files to avoid overwriting features
            csv_file_list = [file for file in csv_file_list if 'combined' not in file]
            if len(csv_file_list) == 0:#the data folder that doesn't have any sessions
                continue
            # sorting to have an ACC followed by an EDA file for each session
            csv_file_list.sort(key=lambda x: x.split(path_style)[-1].split('_')[1])

            # count the number of different files for sessions
            n_files_per_session = len(set([i.split('_')[-1].split('.')[0] for i in csv_file_list]))
            # print("csv_file_list", csv_file_list)
            n_sessions = int(len(csv_file_list)/n_files_per_session)
            print(f"{n_sessions} sessions found for {uid}")

            # process each session in the folder
            for s in range(n_sessions):
                # create a session file list
                session_file_list = csv_file_list[0 + s * n_files_per_session:n_files_per_session + s * n_files_per_session]
                # print(session_file_list)

                list_per_instance = []

                # for file in csv_file_list:
                for file in session_file_list:
                    # print(file)
                    # loading each csv file
                    df = pd.read_csv(file, index_col=None, header=0, dtype=object)
                    try:
                        df_index_time = df.set_index('Timestamp')
                    except KeyError as ke:
                        raise(ke)
                    df_index_time = df_index_time.set_index(pd.to_datetime(df_index_time.index.astype('float'), unit='ms'))
                    match = re.search(r'/(\d+\.\d+)_([\d]{2})_', file)
                    patient_id = match.group(1)
                    session = match.group(2)
                    df_index_time['patient_id'] = patient_id
                    df_index_time['session'] = session

                    # print(df_index_time)

                    list_per_instance.append(df_index_time)

                uid_data_list.append(list_per_instance)
                
        user_dict = {"dataAll": uid_data_list}

        data_dict[uid] = feat_generator(user_dict, bin_size, agg_cat)
        
    return data_dict

def feat_generator(inputDict, bin_size, aggCategory):
    """
    This function generates the features and labels for a given input dictionary.
    
    :param inputDict: dictionary containing the data for a given user
    :param bin_size: int containing the sample period in seconds (e.g., 15)
    :param aggCategory: list with possible aggression labels (e.g., agg_cat = ['AGG', 'SIB', 'ED'])
    :return: dictionary containing the features and labels for a given user
    """
    list_of_sessions = inputDict['dataAll']
    bin_df_per_session = []
    bin_label_per_session = []
    for session in range(len(list_of_sessions)):
        list_per_session = list_of_sessions[session]
        bin_df = pd.DataFrame()
        bin_labels = pd.Series()
        for data_source in range(len(list_per_session)):
            df = list_per_session[data_source]
            df.fillna({'Condition': 0}, inplace=True)
            
            df.loc[df['Condition'].isin(aggCategory), 'Condition'] = 1  # 1 to Agg state

            # Adding the norm of accelerometer data to the data frame.
            if 'X' in df:
                acc_data = (df[['X', 'Y', 'Z']]).to_numpy().astype(np.float)
                df['Magnitude'] = np.linalg.norm(acc_data, axis=1)

            if df['Condition'].nunique() > 2:
                print(df['Condition'].nunique())
                assert ('Have unknown labels!')

            # evidence = list(df.columns.values)
            evidence =  [col for col in df.columns if col != 'patientid_session']
            for tag in ['Condition', 'patient_id', 'session']:
                try:
                    evidence.remove(tag)
                except:
                    pass
            for e in range(len(evidence)):
                bin_df, bin_labels = split_data_into_bins(df=df, evidence=evidence[e], bin_df=bin_df, bin_size=bin_size, bin_labels=bin_labels)
            
            # Generate PPG features
            if 'BVP' in df.columns:
                df['BVP'] = df['BVP'].astype(float)
                df_hr, df_rmssd = gen_ppg_features(df)
                bin_df, bin_labels = split_data_into_bins(df=df_hr, evidence='HR', bin_df=bin_df, bin_size=bin_size, bin_labels=bin_labels)
                bin_df, bin_labels = split_data_into_bins(df=df_rmssd, evidence='RMSSD', bin_df=bin_df, bin_size=bin_size, bin_labels=bin_labels)
            if 'EDA' in df.columns:
                df_phasic_tonic = gen_eda_features(df)
                bin_df, bin_labels = split_data_into_bins(df=df_phasic_tonic, evidence='PHASIC', bin_df=bin_df, bin_size=bin_size, bin_labels=bin_labels)
                bin_df, bin_labels = split_data_into_bins(df=df_phasic_tonic, evidence='TONIC', bin_df=bin_df, bin_size=bin_size, bin_labels=bin_labels)

        proper_order_of_feats = bin_df.columns
        
        # Sets patient session & id as multilevel index
        bin_df['patient_id'] = df['patient_id'].iloc[0]
        bin_df['session'] = df['session'].iloc[0]
        bin_df.set_index(['patient_id', 'session'], append=True, inplace=True)
        bin_df = bin_df.reorder_levels(['patient_id', 'session', bin_df.index.names[0]])
        
        bin_df_per_session.append(bin_df[proper_order_of_feats])    
        bin_label_per_session.append(bin_labels)

    output_dict = {'features': bin_df_per_session, 'labels': bin_label_per_session}
    
    return output_dict

def split_data_into_bins(df, evidence, bin_df, bin_labels, target_fs=16, bin_size=15):
    """
    Resamples one signal to a common frequency, splits into fixed-length time bins,
    and accumulates the result into bin_df. Call once per signal/file.
    The generated labels are the max labels for each bin.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex, a signal column named `evidence`, and a
        'Condition' column.
    evidence : str
        Signal column name to process (e.g. 'BVP', 'EDA', 'X').
    bin_df : pd.DataFrame or None
        Accumulator DataFrame (index = bin timestamps, columns = signal names,
        each cell = np.array of shape (samples_per_bin,)). Pass None on first call.
    bin_labels : pd.Series or None
        Accumulator for per-bin labels (max Condition per bin). Pass None on first
        call; subsequent calls with overlapping sessions will take the max.
    target_fs : int
        Target sampling frequency in Hz.
    bin_size : int
        Duration of each bin in seconds.

    Returns
    -------
    bin_df : pd.DataFrame
        Updated with a new column `evidence`.
    bin_labels : pd.Series
        Per-bin max aggression label.
    """
    samples_per_bin = target_fs * bin_size
    resample_period = f'{1000 // target_fs}ms'

    signal_resampled = df[evidence].astype(float).resample(resample_period).mean().interpolate()
    labels_resampled = df['Condition'].astype(float).resample(resample_period).max().fillna(0)

    n_samples = len(signal_resampled)
    n_bins = n_samples // samples_per_bin
    if n_bins == 0:
        return bin_df, bin_labels

    signal_trimmed = signal_resampled.iloc[:n_bins * samples_per_bin]
    labels_trimmed = labels_resampled.reindex(signal_trimmed.index).fillna(0)

    bin_timestamps = signal_trimmed.index[::samples_per_bin].floor(f'{bin_size}s')
    # Each cell is a 1D array of shape (samples_per_bin,)
    chunks = signal_trimmed.values.reshape(n_bins, samples_per_bin)
    new_col = pd.Series([chunks[i] for i in range(n_bins)], index=bin_timestamps, name=evidence)
    new_col = new_col[~new_col.index.duplicated(keep='first')]

    new_labels = pd.Series(
        labels_trimmed.values.reshape(n_bins, samples_per_bin).max(axis=1),
        index=bin_timestamps
    )
    new_labels = new_labels[~new_labels.index.duplicated(keep='first')]

    if bin_df.empty:
        bin_df = new_col.to_frame()
    else:
        bin_df = bin_df.join(new_col, how='outer')

    if bin_labels.empty:
        bin_labels = new_labels
    else:
        bin_labels = bin_labels.reindex(bin_labels.index.union(new_labels.index)).fillna(0)
        bin_labels = bin_labels.combine(new_labels, max, fill_value=0)

    return bin_df, bin_labels

# ============= Modules for generating instances from binned data =============

def gen_instances_from_raw_feat_dictionary(feat_dict, num_observation_frames, num_prediction_frames, o_multiclass=False,
                                            o_return_list_of_sessions=False, outdir="." ,o_run_from_scratch=False,
                                            bin_size=15):
    """
    This function creates arrays of features and labels in ndarrays
    :param feat_dict:
    :param num_observation_frames:
    :param num_prediction_frames:
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
    selected_feat = [
        'BVP',
        'EDA',
        'X',
        'Y',
        'Z',
        'Magnitude',
        'HR',
        'RMSSD',
        'PHASIC',
        'TONIC',
    ]

    dict_of_instances_arrays = {}
    dict_of_labels_arrays = {}
    dict_of_superposition_lists = {}
    dict_of_session_dfs ={}

    id_blacklist = []

    past_observation_time = num_observation_frames * bin_size
    future_prediction_time = num_prediction_frames * bin_size

    filename = outdir + "/dataInst_to" + str(past_observation_time) + "_tp" \
                + str(future_prediction_time) + "_mc" + str(o_multiclass) + "_rs" + str(o_return_list_of_sessions) \
                + '_bs' + str(bin_size) + 'S.bin'

    if (not o_run_from_scratch) and os.path.isfile(filename):
        # load file
        print('loading data instance data...')
        pickle_in = open(filename, 'rb')
        datalist = pickle.load(pickle_in)
        dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, dict_of_superposition_lists, dict_of_session_dfs = datalist
    else:
        # loops over each subject
        for subject_id in feat_dict:
            dict_of_superposition_lists[subject_id] = []
            try:
                print('subject_id', subject_id)
                
                if o_return_list_of_sessions:
                    instances_array_per_subject_list = []
                    labels_array_per_subject_list = []
                    dfs_array_per_subject_list = []
                else:
                    instances_array_per_subject = np.array([]).reshape(0, num_observation_frames * len(selected_feat) +
                                                                        len(selected_feat))
                    labels_array_per_subject = np.array([]).reshape(0, 1)

                feat_dic_per_subj = feat_dict[subject_id]

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

                    session_instances_array, session_labels_array, signal_cols, instance_df = generate_instances_from_data_bins(
                        feat_data_frame_per_session, 
                        label_data_frame_per_session, 
                        num_observation_frames, 
                        num_prediction_frames, 
                        o_multiclass=o_multiclass)

                    if session_instances_array is None:
                        continue

                    # append list of superposision_index for current
                    sup_list += gen_superposition_index_list(len(session_instances_array), num_observation_frames)

                    if o_return_list_of_sessions:
                        instances_array_per_subject_list.append(session_instances_array)
                        labels_array_per_subject_list.append(session_labels_array)
                        dict_of_superposition_lists[subject_id].append(sup_list)
                        sup_list = []
                        
                        dfs_array_per_subject_list.append(instance_df)
                    
                    else:
                        instances_array_per_subject = np.concatenate((instances_array_per_subject, session_instances_array), axis=0)
                        labels_array_per_subject = np.concatenate((labels_array_per_subject, session_labels_array), axis=0)

                if not o_return_list_of_sessions:
                    dict_of_superposition_lists[subject_id] = sup_list

                if o_return_list_of_sessions:
                    dict_of_instances_arrays[subject_id] = instances_array_per_subject_list
                    dict_of_labels_arrays[subject_id] = labels_array_per_subject_list
                    dict_of_session_dfs[subject_id] = dfs_array_per_subject_list
                else:
                    dict_of_instances_arrays[subject_id] = instances_array_per_subject
                    dict_of_labels_arrays[subject_id] = labels_array_per_subject

            except AssertionError as error:
                print(error)
                id_blacklist.append(subject_id)
                print('Not possible to construct data for sbj ' + subject_id)

        datalist = [dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, dict_of_superposition_lists, dict_of_session_dfs]
        pickle_out = open(filename, 'wb')
        pickle.dump(datalist, pickle_out)

    return dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, dict_of_superposition_lists, signal_cols, dict_of_session_dfs

def gen_ppg_features(df, fs=64, preprocessed=False, window_size_rmssd=30, step_size_rmssd=1):   
    """
    Generate dataframes for HR and RMSSD.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex, a signal column named `evidence`, and a
        'Condition' column.
    fs : int
        Sampling frequency in Hz.
    preprocessed : bool
        Whether the data is preprocessed (filtered) or not.
        
    Returns
    -------
    df_hr : pd.DataFrame
        DataFrame with a DatetimeIndex, a column named `HR`, and a 'Condition' column.
    df_rmssd : pd.DataFrame
        DataFrame with a DatetimeIndex, a column named `RMSSD`, and a 'Condition' column.
    """
    df = df.copy()
    df = df.reset_index()

    if not preprocessed:
        filt = PPG.Filters(fs=fs)
        filtered = filt.filter_signal(df['BVP'], lowcut=0.5, highcut=4.0, order=4, window_len=0.5)
        df['BVP_filtered'] = filtered

    bvp_col = 'BVP_filtered'
    if bvp_col not in df.columns:
        raise ValueError(f"Column {bvp_col} not found in dataframe")

    # Detect heartbeats in BVP
    beat_detector = PPG.BeatDetectors(fs=fs, preprocessed=True)
    peaks = beat_detector.adaptive_threshold(df[bvp_col])
    df.loc[peaks, 'Peak'] = 1

    # Identify artifactual beats
    cardio = SQA.Cardio(fs=fs)
    beats_ix = df.loc[df['Peak'] == 1].index
    artifacts_ix = cardio.identify_artifacts(beats_ix=beats_ix, method='both')
    df['Artifact'] = np.nan
    df.loc[artifacts_ix, 'Artifact'] = 1

    # Get instantaneous heart rate
    hr, peak_idx_for_hr, timestamps_hr = pp.get_instantaneous_heart_rate(
        pd.Series(df.loc[df['Peak'] == 1].index), 
        fs=fs, beat_timestamps=df.loc[df['Peak'] == 1, 'Timestamp']
    )
    artifactual_hr_i = [i for i, idx in enumerate(peak_idx_for_hr) if idx in artifacts_ix]
    hr_filtered = deepcopy(hr)
    hr_filtered[artifactual_hr_i] = np.nan

    # interpolate nan values
    hr_interpolated = np.interp(np.where(np.isnan(hr_filtered))[0], np.where(~np.isnan(hr_filtered))[0], hr_filtered[~np.isnan(hr_filtered)])
    hr_filtered[np.where(np.isnan(hr_filtered))[0]] = hr_interpolated
    # Smooth the HR signal
    hr_smoothed = pp.moving_average(hr_filtered, 20)
    # Combine with the original data frame
    df_hr = pd.DataFrame({
        'Timestamp': timestamps_hr,
        'HR': hr_smoothed,
        'Condition': df['Condition'].reindex(peak_idx_for_hr).fillna(0).reset_index(drop=True)
    })

    # Get RMSSD
    rmssd, rmssd_timestamps = pp.get_rmssd(beats_ix, fs, window_size_rmssd, step_size_rmssd, artifacts_ix=artifacts_ix, beat_timestamps=df.loc[beats_ix, 'Timestamp'])
    # interpolate nan values
    rmssd_interpolated = np.interp(np.where(np.isnan(rmssd))[0], np.where(~np.isnan(rmssd))[0], rmssd[~np.isnan(rmssd)])
    rmssd[np.where(np.isnan(rmssd))[0]] = rmssd_interpolated
    # Smooth the RMSSD signal
    rmssd_smoothed = pp.moving_average(rmssd, 3)
    # Create RMSSD dataframe
    df = df.set_index('Timestamp')
    df_rmssd = pd.DataFrame(
        {
            'Timestamp': rmssd_timestamps, 
            'RMSSD': rmssd_smoothed,
            'Condition': df['Condition'].reindex(rmssd_timestamps).fillna(0)
        }
    )


    df_hr = df_hr.set_index('Timestamp')
    df_rmssd = df_rmssd.set_index('Timestamp')

    return df_hr, df_rmssd

def gen_eda_features(df, fs=4, preprocessed=False):
    """
    Generate dataframes for PHASIC and TONIC.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex, a signal column named `evidence`, and a
        'Condition' column.
    fs : int
        Sampling frequency in Hz.
    preprocessed : bool
        Whether the data is preprocessed (filtered) or not.
        
    Returns
    -------
    df_phasic_tonic : pd.DataFrame
        DataFrame with a DatetimeIndex, a column named `PHASIC`, a column named `TONIC`, and a 'Condition' column.
    """
    df = df.copy()
    # Filter EDA signal
    if not preprocessed:
        filt = EDA.Filters(fs=fs)
        filtered = filt.lowpass_gaussian(df['EDA'])
        df['EDA_filtered'] = filtered
    
    eda_col = 'EDA_filtered'
    if eda_col not in df.columns:
        raise ValueError(f"Column {eda_col} not found in dataframe")
    
    record_duration = (df.index[-1] - df.index[0]).total_seconds()
    record_length = len(df)
    fs = round(record_length / record_duration, 0)
    df[eda_col] = df[eda_col].astype(float)
    phasic, tonic = EDA.decompose_eda(df[eda_col], fs=fs)
    df_phasic_tonic = pd.DataFrame({
        'Timestamp': df.index,
        'PHASIC': phasic,
        'TONIC': tonic,
        'Condition': df['Condition']
    })
    df_phasic_tonic = df_phasic_tonic.set_index('Timestamp')
    return df_phasic_tonic

def generate_instances_from_data_bins(bin_df, bin_labels, n_obs_bins=12, n_pred_bins=12, o_multiclass=False):
    """
    Combines consecutive 15s bins into observation windows and assigns future labels.
    Returns bin_start_indices so train/test splits can avoid overlapping windows.

    Parameters
    ----------
    bin_df : pd.DataFrame
        Index = bin timestamps, columns = signal names, each cell = np.array of
        shape (samples_per_bin,). Output of repeated split_data_into_bins() calls.
    bin_labels : pd.Series
        Per-bin max aggression label. Index must match bin_df.
    n_obs_bins : int
        Number of bins per observation window (e.g. 12 × 15s = 3 min).
    n_pred_bins : int
        Number of future bins used for label assignment (e.g. 12 × 15s = 3 min).
    o_multiclass : bool
        Whether to include the multiclass labels in the data.
        If True, the data will be returned as a multiclass model.
        If False, the data will be returned as a binary model.
    Returns
    -------
    instances : np.ndarray, shape (n_instances, n_channels, n_obs_bins * samples_per_bin)
        Each instance is n_obs_bins consecutive bins concatenated along the time axis.
    labels : np.ndarray, shape (n_instances,)
        1 if any aggression occurs in the future prediction window, else 0.
    signal_cols : list of str
        List of signal columns in the order they appear in the dataframe.
    instance_df : pd.DataFrame
    """
    label_values = bin_labels.reindex(bin_df.index).fillna(0).values
    signal_cols = list(bin_df.columns)
    n_bins = len(bin_df)
    instances, labels = [], []

    for i in range(n_bins - n_obs_bins - n_pred_bins + 1):
        if label_values[i:i + n_obs_bins].max() > 0:
            continue  # skip: aggression already ongoing in observation window

        # Skip windows where any signal has missing bins (NaN from outer join)
        window = bin_df.iloc[i:i + n_obs_bins]
        if window.applymap(lambda v: not isinstance(v, np.ndarray)).any().any():
            continue

        # Stack signals: (n_channels, n_obs_bins * samples_per_bin)
        instance = np.stack([
            np.concatenate(window[col].values)
            for col in signal_cols
        ])
        if o_multiclass:
            label = label_values[i + n_obs_bins:i + n_obs_bins + n_pred_bins]
        else:
            label = int(label_values[i + n_obs_bins:i + n_obs_bins + n_pred_bins].max() > 0)

        instances.append(instance)
        labels.append(label)
    
    instance_df = pd.DataFrame(instances, columns=signal_cols)

    if not instances:
        return np.array([]), np.array([]), signal_cols, pd.DataFrame()

    return np.stack(instances), np.array(labels), signal_cols, instance_df

# ============= Modules for generating superposition indices =============

def gen_superposition_index_list(num_of_instances, num_observation_frames):
    """
    Generates a list of superposition indices for each instance.
    The superposition indices are the indices of the instances that overlap with the current instance.
    The superposition indices are used to enforce non-overlapping train/test splits.
    
    Parameters
    :param num_of_instances: number of instances
    :param num_observation_frames: number of observation frames
    :return: list of superposition indices
    """
    sup_list = []
    for i in range(num_of_instances):
        sup_list.append([i - max(0, i - num_observation_frames + 1),
                        min(i + num_observation_frames - 1, num_of_instances) - i])
    return sup_list

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dp", "--data_path", type=str, default='../../CBS_DATA_ASD_ONLY', help="Data path")
    parser.add_argument("-bs", "--bin_size", type=int, default=15, help="Bin size in seconds")
    parser.add_argument("-ac", "--agg_cat", type=list, default=['AGG','SIB', 'ED'], help="Aggression categories")
    parser.add_argument("-ofr", "--o_run_from_scratch", action='store_true', help="Run from scratch")
    parser.add_argument("-mc", "--o_multiclass", action='store_true', help="Multiclass")
    parser.add_argument("-rol", "--o_return_list_of_sessions", action='store_true', help="Return list of sessions")
    parser.add_argument("-no", "--num_observation_frames", type=int, default=12, help="Number of observation frames")
    parser.add_argument("-np", "--num_prediction_frames", type=int, default=4, help="Number of prediction frames")
    
    args = parser.parse_args()
    data_path = args.data_path
    bin_size = args.bin_size
    agg_cat = args.agg_cat
    o_run_from_scratch = args.o_run_from_scratch
    o_multiclass = args.o_multiclass
    o_return_list_of_sessions = args.o_return_list_of_sessions
    num_observation_frames = args.num_observation_frames
    num_prediction_frames = args.num_prediction_frames

    dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, uid_dict, \
        dict_of_superposition_lists, feat_col_names, dict_of_session_dfs = data_preprocess(
            data_path=data_path, num_observation_frames=num_observation_frames, num_prediction_frames=num_prediction_frames,
            o_run_from_scratch=o_run_from_scratch, o_return_list_of_sessions=o_return_list_of_sessions,  
            bin_size=bin_size, o_multiclass=o_multiclass, 
        )
