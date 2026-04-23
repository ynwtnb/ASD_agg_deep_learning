import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, IntSlider, fixed

def get_df_raw_data_combined(base_path, t, patient_id,session_num):
    """
    Load raw data (combined) from a CSV file based on provided parameters, convert 'Timestamp' column to datetime, 
    and validate for NaT values.

    Parameters:
        base_path (str): Base directory path for the data.
        t (str): Specific folder or category under base_path.
        patient_id (str): Unique identifier for the patient.
        session_num (str): Session number of the data file.

    Returns:
        pd.DataFrame: DataFrame with data from the CSV file, 'Timestamp' converted to datetime.

    Raises:
        ValueError: If any NaT values are present in the 'Timestamp' column.
    """
    path = f'{base_path}/{t}/{patient_id}/{patient_id}_{session_num}_combined_matched.csv'

    df_raw = pd.read_csv(path)
            
    df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], unit='ms')
    if df_raw['Timestamp'].isnull().any():
        raise ValueError("There are NaT (Not a Time) values in the 'Timestamp' column after conversion.")
    else:
        return df_raw

def get_df_raw_data_modality(base_path, t, patient_id,session_num, modality):
    """
    Load raw data (per modality) from a CSV file based on provided parameters, convert 'Timestamp' column to datetime, 
    and validate for NaT values.

    Parameters:
        base_path (str): Base directory path for the data.
        t (str): Specific folder or category under base_path.
        patient_id (str): Unique identifier for the patient.
        session_num (str): Session number of the data file.
        modality (str): Modality of the data file. 'BVP', 'ACC', or 'EDA.'

    Returns:
        pd.DataFrame: DataFrame with data from the CSV file, 'Timestamp' converted to datetime.

    Raises:
        ValueError: If any NaT values are present in the 'Timestamp' column.
    """
    path = f'{base_path}/{t}/{patient_id}/{patient_id}_{session_num}_{modality}_matched.csv'

    df_raw = pd.read_csv(path)
            
    df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], unit='ms')
    if df_raw['Timestamp'].isnull().any():
        raise ValueError("There are NaT (Not a Time) values in the 'Timestamp' column after conversion.")
    else:
        return df_raw

# scales better for viz
def replace_for_better_viz(df, default_tpa = 5000.0, replacement_default_tpa=200, AOF_true = 1.0, replacement_AOF_true=150):
    cols = df.columns
    tpa_col = [col for col in cols if 'TimePastAggression' in col]
    df[tpa_col] = df[tpa_col].replace(default_tpa,replacement_default_tpa)

    aof_col = [col for col in cols if 'AGGObserved' in col]
    df[aof_col] = df[aof_col].replace(AOF_true,replacement_AOF_true)

    return df


def filter_df_cols(df, col_group, exclude_std_dev=False):
    acc_cols = [col for col in df.columns if 'ACC' in col]
    eda_cols = [col for col in df.columns if 'EDA' in col]
    bvp_cols = [col for col in df.columns if 'BVP' in col]
    
    time_based_cols = [col for col in df.columns if 'AGGObserved' in col or 'TimePastAggression' in col]
    
    raw_label_col = [col for col in df.columns if 'rawLabel' in col]
    processed_label_col = [col for col in df.columns if 'processedLabel' in col]
    
    physio_cols = acc_cols+eda_cols+bvp_cols
    feature_cols = physio_cols + time_based_cols
    label_cols = raw_label_col + processed_label_col
    pred_col = [col for col in df.columns if 'predict_proba' in col]
    
    col_groups = {
        'acc_cols':acc_cols,
        'eda_cols':eda_cols,
        'bvp_cols':bvp_cols,
        'time_based_cols':time_based_cols,
        'raw_label_col':raw_label_col,
        'processed_label_col':processed_label_col,
        'physio_cols':physio_cols,
        'feature_cols':feature_cols,
        'label_cols':label_cols,
        'pred_col':pred_col,        
    }
    
    if exclude_std_dev and col_group in ['acc_cols','eda_cols','bvp_cols','time_based_cols']:
        cols = [col for col in col_groups[col_group] if 'window_std_dev' not in col]
        return df[cols]
    else:
        return df[col_groups[col_group]]    

def filter_df_for_session(df_processed, patient_id, session, summary_stat='mean'):
    """
    Filter processed data for a specific patient session, selecting columns with specified summary statistics.
    Always extracts t-15s bin.

    Parameters:
        df_processed (pd.DataFrame): DataFrame containing processed data.
        patient_id (str): Unique identifier for the patient.
        session (str): Session number to filter.
        summary_stat (str, optional): Summary statistic suffix to filter columns (default is 'mean').

    Returns:
        pd.DataFrame: Filtered DataFrame containing only relevant columns and rows for the specified patient and session.
    """

    summary_stat_cols = [col for col in df_processed.columns if col.endswith(f"{summary_stat}_t-15s")]
    time_based_cols = ['AGGObserved_t-15s','TimePastAggression_t-15s']
    label_cols = [col for col in df_processed.columns if 'Label' in col]
    pred_col = [col for col in df_processed.columns if 'predict_proba' in col]

    selected_cols = summary_stat_cols + time_based_cols + label_cols + pred_col
    
    df_filtered = df_processed[selected_cols]
    idx = pd.IndexSlice
    df_filtered = df_filtered.loc[idx[patient_id, session, :], :]
    
    return df_filtered


def plot_patient_data_raw_and_processed_overlay_plotly(
        patient_id,
        session,
        df_raw,
        df_processed,
        summary_stat='mean',
        start_time_shift_minutes=0,
        time_window_minutes=None,
        X=['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'EDA'],
        Y=['AGG', 'ED', 'SIB'],
        normalize=False,
        start_time_from_0 = False,
        dotted_lines_every_15s=False
        ):

    # if start_time_from_0:
    #     # Subtract the minimum timestamp from all timestamps to start from 0
    #     df_raw['Timestamp'] = (df_raw['Timestamp'] - df_raw['Timestamp'].min()).astype('timedelta64[ns]')

    if normalize and isinstance(df_processed,pd.DataFrame):
        df_raw[X] = (df_raw[X] - df_raw[X].mean()) / df_raw[X].std()
        feature_columns = [col for col in df_processed.columns if ('Label' not in col) and ('predict_proba' not in col)]
        df_processed[feature_columns] = (df_processed[feature_columns] - df_processed[feature_columns].mean()) / df_processed[feature_columns].std()

    # start_time = df['Timestamp'].min() + pd.Timedelta(minutes=start_time_shift_minutes)
    start_time = df_raw['Timestamp'].iloc[0]    
    if time_window_minutes:
        end_time = start_time + pd.Timedelta(minutes=time_window_minutes)
    else:
        # end_time = df['Timestamp'].max()
        end_time = df_raw['Timestamp'].iloc[-1]
    
    feature_color_mapping = {
        'ACC_X': 'blue',
        'ACC_Y': 'green',
        'ACC_Z': 'purple',
        'BVP': 'orange',
        'EDA': 'magenta',
        'TimePastAggression': 'brown',
        'AGGObserved': 'gray'
    }
    event_color_mapping = {
        'AGG': 'red',
        'ED': 'blue',
        'SIB': 'orange'
    }
    aggression_label_color_mapping = {
        1: 'blue',
        2: 'orange',
        3: 'red'
    }

    df_raw = df_raw[(df_raw['Timestamp'] >= start_time) & (df_raw['Timestamp'] <= end_time)]

    # Create the figure
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Plot the raw data
    for x in X:
        fig.add_trace(go.Scatter(x=df_raw['Timestamp'],
                                 y=df_raw[x],
                                 mode='lines',
                                 name=x,
                                 opacity=0.5,
                                 line=dict(color=feature_color_mapping.get(x, 'black'))
                                 )
        )
    
    # plot the raw labels
    for y in Y:
        occurrences = df_raw.loc[df_raw[y] == 1, 'Timestamp']

        if not occurrences.empty:
            color = event_color_mapping.get(y, 'green')
            
            # Duplicate each timestamp and add np.nan in between
            x_values = []
            for occurrence in occurrences:
                x_values.extend([occurrence, occurrence, np.nan])  # add np.nan after each occurrence pair
            
            y_min = 0#df_raw[X].min().min()
            y_max = df_raw[X].max().max()
            
            # Repeat the y-range and add np.nan in between for gaps
            y_values = []
            for _ in occurrences:
                y_values.extend([y_min, y_max, np.nan])

            fig.add_trace(go.Scatter(
                x=x_values,  # Use x_values with np.nan to create gaps
                y=y_values,  # Use y_values with np.nan to create gaps
                mode='lines',
                line=dict(color=color),
                name=y,
                opacity=0.4,
                showlegend=True
            ))


    if isinstance(df_processed,pd.DataFrame):
        # time_based_features = [col for col in df_processed.columns if ('TimePastAggression' in col) or ('AGGObserved' in col)]
        time_based_features = ['TimePastAggression','AGGObserved']
        measurements_to_plot = X + time_based_features + ['predict_proba']
        
        # plot processed features
        for x in measurements_to_plot:
            summary_stat = summary_stat if x not in time_based_features else ''
            time_indicator = '_t-15s' if x != 'predict_proba' else ''
            
            if x == 'predict_proba':
                fig.add_trace(go.Scatter(x=df_processed.index.get_level_values(2).unique().tolist(),
                                    y=df_processed[x],
                                    mode='lines',
                                    name=x,
                                    opacity=0.5,
                                    line=dict(color=feature_color_mapping.get(x, 'black'))
                                    ))
            
                continue
            
            if f"{x}{summary_stat}{time_indicator}" in df_processed.columns:
                summary_stat = summary_stat if x not in time_based_features else ''
                fig.add_trace(go.Scatter(
                    x=df_processed.index.get_level_values(2).unique().tolist(),
                    y=df_processed[f"{x}{summary_stat}{time_indicator}"],
                    mode='markers',  # Change 'lines' to 'markers' to show points
                    name=f'{x}{" "+summary_stat if summary_stat else ""}',
                    marker=dict(color=feature_color_mapping.get(x, 'black'), symbol='circle', size=6),
                    opacity=0.8
                ))
        
        # Mapping label values to aggression types for legend
        aggression_label_name_mapping = {
            1: 'ED',
            2: 'SIB',
            3: 'AGG'
        }


        raw_label_col = [col for col in df_processed.columns if 'rawLabel' in col]
        raw_label_col = raw_label_col[0]
        
        processed_label_col = [col for col in df_processed.columns if 'processedLabel' in col]
        processed_label_col = processed_label_col[0]

        # Replace 0 values with NaN in raw labels column
        df_processed[raw_label_col] = df_processed[raw_label_col].replace(0, np.nan)

        
        # Add scatter points for each aggressionLabel value with color mapping and legend names
        for label_value, color in aggression_label_color_mapping.items():
            # Filter only rows with the specific aggressionLabel value
            label_occurrences = df_processed[df_processed[processed_label_col] == label_value]
            
            label_multiplier = 300 if not normalize else df_raw[X].max().max()
            label_y_multiplier = df_raw[X].max().max()
            if not label_occurrences.empty:
                fig.add_trace(go.Scatter(
                    x=label_occurrences.index.get_level_values(2).unique().tolist(),
                    y=label_occurrences[processed_label_col]/label_occurrences[processed_label_col] * label_y_multiplier,  # Offset to separate from main plot
                    mode='markers',
                    name=f"{aggression_label_name_mapping[label_value]}_processed_label",  # Map value to name for legend
                    marker=dict(color=color, symbol='circle', size=6),
                    opacity=0.4
                ))

    if dotted_lines_every_15s:
        # Add vertical lines every 15 seconds within the time window (these won't be in the legend)
        current_time = start_time
        while current_time <= end_time:
            current_time_datetime = current_time  # Convert to native datetime
            fig.add_vline(x=current_time_datetime, line=dict(color="gray", width=0.5, dash="dash"), opacity=0.5)
            current_time += pd.Timedelta(seconds=15)

    # Format the x-axis to show only the time of day
    fig.update_xaxes(tickformat='%H:%M:%S')

    # Update labels
    fig.update_layout(
        title=f'Raw Aggression Data - Patient_ID: {patient_id}, Session: {session}',
        xaxis_title='Time',
        yaxis_title='Physiological Activity',
        legend_title="Legend",
        plot_bgcolor='white',
        width=1350,
        height=1000,
        xaxis_range=[start_time, end_time]
    )
    print('Session start:',start_time)
    print('Session end:',end_time)
    print('Elapsed time:',end_time-start_time)
    # return fig
    fig.show()

def update_interactive_plot(df, window_size, start_window, cols, show_ppg_peaks=False, show_artifacts=False):
    """
    Update the interactive plot based on the selected window size and start window.
    Parameters:
        df (pd.DataFrame): DataFrame containing the raw data to be plotted.
        window_size (int): Size of the time window in seconds.
        start_window (int): Starting window index for the plot.
        cols (list): List of columns to be plotted.
    """
    color_map = {'AGG': 'tomato', 'ED': 'skyblue', 'SIB': 'orange'}
    start_time = df['Timestamp'].min() + pd.Timedelta(seconds=window_size * start_window)
    end_time = start_time + pd.Timedelta(seconds=window_size)
    df_window = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] < end_time)]
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    legends = []
    legend_labels = []
    
    # Plot data columns
    for i, col in enumerate(cols):
        if i != 0:
            twinx = ax.twinx()
            legend, = twinx.plot(df_window['Timestamp'], df_window[col], label=col, color=f'C{i}')
        else:    
            legend, = ax.plot(df_window['Timestamp'], df_window[col], label=col, color=f'C{i}')
        legends.append(legend)
        legend_labels.append(col)
    
    # Find the PPG signal column (BVP or Filtered, whichever is in cols)
    ppg_col_name = next((col for col in cols if col == 'Filtered' or 'BVP' in col), None)

    # Handle PPG peaks
    if show_ppg_peaks and ppg_col_name:
        ppg_peaks = df_window[df_window['Peak'] == 1]
        legend = ax.scatter(x=ppg_peaks['Timestamp'], y=ppg_peaks[ppg_col_name], color='gray', label='Peak')
        legends.append(legend)
        legend_labels.append('Peak')

    # Handle artifacts
    if show_artifacts and ppg_col_name:
        artifacts = df_window[df_window['Artifact'] == 1]
        legend = ax.scatter(x=artifacts['Timestamp'], y=artifacts[ppg_col_name], color='gold', label='Artifact')
        legends.append(legend)
        legend_labels.append('Artifact')
    
    # Handle aggression events
    for agg in ['AGG', 'ED', 'SIB']:
        positive_events = df_window[df_window[agg] == 1]
        agg_legend_added = False
        for _, row in positive_events.iterrows():
            legend = ax.axvline(x=row['Timestamp'], color=color_map[agg], label=agg, alpha=0.1)
            if not agg_legend_added:
                legends.append(legend)
                legend_labels.append(agg)
                agg_legend_added = True
    
    fig.legend(legends, legend_labels, loc='upper right')
    plt.show()

def interactive_plot(df, window_size, cols, show_ppg_peaks=False, show_artifacts=False):
    """
    Create an interactive plot for the raw data with a specified window size and start window.
    Parameters:
        df (pd.DataFrame): DataFrame containing the raw data to be plotted.
        window_size (int): Size of the time window in seconds.
        cols (list): List of columns to be plotted.
    """
    assert not show_ppg_peaks or 'Peak' in df.columns, "Peaks are only shown if 'Peak' column is present in the dataframe."
    assert not show_ppg_peaks or any('BVP' in col or col == 'Filtered' for col in cols), "Peaks are only shown if a BVP or Filtered column is present in the cols list."
    assert not show_artifacts or 'Artifact' in df.columns, "Artifacts are only shown if 'Artifact' column is present in the dataframe."
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    duration = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()
    n_windows = duration // window_size + 1
    interact(update_interactive_plot, df=fixed(df), window_size=fixed(window_size), start_window=IntSlider(min=0, max=n_windows, step=1), cols=fixed(cols), show_ppg_peaks=fixed(show_ppg_peaks), show_artifacts=fixed(show_artifacts))


def update_interactive_plot_metrics(df, metrics, window_size, start_window, cols):
    """
    Update the interactive plot based on the selected window size and start window.
    Parameters:
        df (pd.DataFrame): DataFrame containing the raw data to be plotted.
        window_size (int): Size of the time window in seconds.
        start_window (int): Starting window index for the plot.
        cols (list): List of columns to be plotted.
    """
    color_map = {'AGG': 'tomato', 'ED': 'skyblue', 'SIB': 'orange'}
    start_time = df['Timestamp'].min() + pd.Timedelta(seconds=window_size * start_window)
    end_time = start_time + pd.Timedelta(seconds=window_size)
    df_window = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] < end_time)]
    metrics_window = metrics[(metrics['Timestamp'] >= start_time) & (metrics['Timestamp'] < end_time)]
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    legends = []
    legend_labels = []
    
    # Plot data columns
    for i, col in enumerate(cols):
        if i != 0:
            twinx = ax.twinx()
            legend, = twinx.plot(metrics_window['Timestamp'], metrics_window[col], label=col, color=f'C{i}')
            twinx.set_ylabel(col)
        else:    
            legend, = ax.plot(metrics_window['Timestamp'], metrics_window[col], label=col, color=f'C{i}')
            ax.set_ylabel(col)
        legends.append(legend)
        legend_labels.append(col)
    
    # Handle aggression events
    for agg in ['AGG', 'ED', 'SIB']:
        positive_events = df_window[df_window[agg] == 1]
        agg_legend_added = False
        for _, row in positive_events.iterrows():
            legend = ax.axvline(x=row['Timestamp'], color=color_map[agg], label=agg, alpha=0.1)
            if not agg_legend_added:
                legends.append(legend)
                legend_labels.append(agg)
                agg_legend_added = True
    
    fig.legend(legends, legend_labels, loc='upper right')

def interactive_plot_metrics(df, metrics, window_size, cols):
    """
    Create an interactive plot for the raw data with a specified window size and start window.
    Parameters:
        df (pd.DataFrame): DataFrame containing the raw data to be plotted.
        metrics (pd.DataFrame): DataFrame containing the metrics to be plotted.
        window_size (int): Size of the time window in seconds.
        cols (list): List of columns to be plotted.
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    duration = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()
    n_windows = duration // window_size + 1
    interact(update_interactive_plot_metrics, df=fixed(df), metrics=fixed(metrics), window_size=fixed(window_size), start_window=IntSlider(min=0, max=n_windows, step=1), cols=fixed(cols))

def plot_instantaneous_heart_rate(df):
    """
    Plot the instantaneous heart rate from the PPG signal.
    Parameters:
        df (pd.DataFrame): DataFrame containing the raw data to be plotted. Must have a 'Timestamp' column and a 'HR' column.
        fs (int): Sampling frequency of the PPG signal.
    """
    event_color_mapping = {
        'AGG': 'red',
        'ED': 'blue',
        'SIB': 'orange'
    }
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df_peak = df.copy().dropna(subset=['HR'])
    timestamp = df_peak['Timestamp']
    hr = df_peak['HR']
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=timestamp, y=hr, mode='lines', name='HR', opacity=0.5, line=dict(color='black')))
    fig.update_layout(
        title='Instantaneous Heart Rate',
        xaxis_title='Time',
        yaxis_title='Heart Rate',
    )
    agg_labels = ['AGG', 'ED', 'SIB']
    for agg in agg_labels:
        agg_occurrences = df[df[agg] == 1]
        if not agg_occurrences.empty:
            agg_time_diff = agg_occurrences.index.diff()
            mask_onset = (agg_time_diff > 1) | (agg_time_diff.isna())
            agg_onset = agg_occurrences[mask_onset].index
            agg_time_diff_inv = agg_occurrences.index.diff(periods=-1)
            mask_offset = (agg_time_diff_inv < -1) | (agg_time_diff_inv.isna())
            agg_offset = agg_occurrences[mask_offset].index

            for onset, offset in zip(agg_onset, agg_offset):
                fig.add_vrect(
                    x0=agg_occurrences.loc[onset, 'Timestamp'],
                    x1=agg_occurrences.loc[offset, 'Timestamp'],
                    fillcolor=event_color_mapping[agg],
                    opacity=0.5
                )
    fig.show()

def plot_rmssd(df_rmssd, df_agg_labels):
    """
    Plot the RMSSD from the PPG signal.
    Parameters:
        df (pd.DataFrame): DataFrame containing the raw data to be plotted. Must have a 'Timestamp' column and a 'RMSSD' column.
    """
    event_color_mapping = {
        'AGG': 'red',
        'ED': 'blue',
        'SIB': 'orange'
    }
    df_rmssd = df_rmssd.copy()
    df_rmssd['Timestamp'] = pd.to_datetime(df_rmssd['Timestamp'])
    df_agg_labels = df_agg_labels.copy()
    df_agg_labels['Timestamp'] = pd.to_datetime(df_agg_labels['Timestamp'])
    timestamp = df_rmssd['Timestamp']
    rmssd = df_rmssd['RMSSD']
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=timestamp, y=rmssd, mode='lines', name='RMSSD', opacity=0.5, line=dict(color='black')))
    fig.update_layout(
        title='RMSSD',
        xaxis_title='Time',
        yaxis_title='RMSSD',
    )
    agg_labels = ['AGG', 'ED', 'SIB']
    for agg in agg_labels:
        agg_occurrences = df_agg_labels[df_agg_labels[agg] == 1]
        if not agg_occurrences.empty:
            agg_time_diff = agg_occurrences.index.diff()
            mask_onset = (agg_time_diff > 1) | (agg_time_diff.isna())
            agg_onset = agg_occurrences[mask_onset].index
            agg_time_diff_inv = agg_occurrences.index.diff(periods=-1)
            mask_offset = (agg_time_diff_inv < -1) | (agg_time_diff_inv.isna())
            agg_offset = agg_occurrences[mask_offset].index

            for onset, offset in zip(agg_onset, agg_offset):
                fig.add_vrect(
                    x0=agg_occurrences.loc[onset, 'Timestamp'],
                    x1=agg_occurrences.loc[offset, 'Timestamp'],
                    fillcolor=event_color_mapping[agg],
                    opacity=0.5
                )
    fig.show()