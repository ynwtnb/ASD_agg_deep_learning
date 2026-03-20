import pandas as pd
from scipy.signal import cheby2, filtfilt
import numpy as np

def filter_ppg(ppg: pd.Series, fs: int, low: float, high: float, order: int) -> pd.Series:
    '''
    A function to filter the PPG signal using a Chebyshev Type II filter.

    Parameters
    ---
    ppg: pd.Series
        The Series containing the PPG signal.
    fs: int
        The sampling frequency of the PPG signal.
    low: float
        The lowcut frequency of the filter.
    high: float
        The highcut frequency of the filter.
    order: int
        The order of the filter.
    
    Returns
    ---
    filtered: pd.Series
        The Series containing the filtered PPG signal. 
    '''
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    b, a = cheby2(order, 20, Wn = [low, high], btype = 'bandpass')
    filtered = filtfilt(b, a, ppg)
    return filtered

def get_instantaneous_heart_rate(peak_idx: pd.Series, fs: int,
                                  beat_timestamps: pd.Series = None) -> tuple:
    '''
    A function to calculate the instantaneous heart rate from the PPG signal.

    Parameters
    ---
    peak_idx: pd.Series
        The Series containing the indices of the peaks in the PPG signal.
    fs: int
        The sampling frequency of the PPG signal.
    beat_timestamps: pd.Series, optional
        Timestamps corresponding to each peak in peak_idx. If provided, the returned
        timestamps will correspond to each HR value (i.e. the second peak of each IBI).

    Returns
    ---
    instantaneous_heart_rate: np.ndarray
        The instantaneous heart rate values in BPM.
    peak_idx_for_hr: pd.Series
        The peak indices corresponding to each HR value (second peak of each IBI).
    timestamps: pd.Series or None
        Timestamps for each HR value. Only returned if beat_timestamps is provided.
    '''
    instantaneous_heart_rate = 60 / (np.diff(peak_idx) / fs)
    peak_idx_for_hr = peak_idx[1:]
    if beat_timestamps is not None:
        timestamps = beat_timestamps.iloc[1:].reset_index(drop=True)
        return instantaneous_heart_rate, peak_idx_for_hr, timestamps
    return instantaneous_heart_rate, peak_idx_for_hr

def get_rmssd(peak_idx: np.ndarray, fs: int, window_size: int, window_step: int,
              beat_timestamps: np.ndarray = None,
              artifacts_ix: np.ndarray = None) -> tuple:
    '''
    A function to calculate the root mean square of successive differences (RMSSD) from the PPG signal.

    Parameters
    ---
    peak_idx: np.ndarray
        The indices of the peaks in the PPG signal.
    fs: int
        The sampling frequency of the PPG signal.
    window_size: int
        The size of the window to calculate the RMSSD in seconds.
    window_step: int
        The step size of the window to calculate the RMSSD in seconds.
    beat_timestamps: np.ndarray, optional
        Timestamps corresponding to each peak in peak_idx (e.g. pd.DatetimeIndex or
        array of datetime objects). If provided, the returned timestamps will be in the
        same format, anchored to the first beat and offset by window_size per window.
        If None, timestamps are returned as elapsed seconds from the first beat.
    artifacts_ix: np.ndarray, optional
        Indices of artifactual beats. If provided, any IBI where either bounding
        peak is artifactual will be set to NaN and excluded from RMSSD computation.

    Returns
    ---
    rmssd: np.ndarray
        The RMSSD values for each window.
    timestamps: np.ndarray
        The timestamp at the end of each window. dtype matches beat_timestamps if
        provided, otherwise float seconds from the first beat.
    '''
    import pandas as pd

    peak_idx = np.asarray(peak_idx)

    # Calculate IBI (Inter-Beat Intervals) in seconds
    ibi = np.diff(peak_idx) / fs

    # Mask IBIs that involve an artifactual beat
    if artifacts_ix is not None:
        artifacts_set = set(artifacts_ix)
        for i in range(len(ibi)):
            if peak_idx[i] in artifacts_set or peak_idx[i + 1] in artifacts_set:
                ibi[i] = np.nan

    # Calculate successive differences of IBI (NaN propagates naturally via np.diff)
    ibi_diff = np.diff(ibi)
    # Insert NaN at the beginning to match the length of ibi
    ibi_diff = np.insert(ibi_diff, 0, np.nan)

    # Use actual beat times for windowing so NaN IBIs don't corrupt the time axis
    ibi_times = (peak_idx[1:] - peak_idx[0]) / fs

    # Calculate total duration
    total_duration = (peak_idx[-1] - peak_idx[0]) / fs

    # Calculate number of windows
    if window_step <= 0:
        raise ValueError("window_step must be positive")

    n_windows = max(0, int((total_duration - window_size) // window_step) + 1)

    if n_windows <= 0:
        return np.array([]), np.array([])

    rmssd = np.zeros(n_windows)

    for i in range(n_windows):
        start_time = i * window_step
        end_time = start_time + window_size

        # Find IBI differences that fall within this window
        window_mask = (ibi_times >= start_time) & (ibi_times < end_time)

        # Get the corresponding IBI differences (NaN values excluded by nanmean)
        window_ibi_diff = ibi_diff[window_mask]

        if len(window_ibi_diff) > 0:
            # Calculate RMSSD: root mean square of successive differences
            rmssd[i] = np.sqrt(np.nanmean((window_ibi_diff * 1000) ** 2))
        else:
            rmssd[i] = np.nan

    # Compute timestamps at the end of each window
    if beat_timestamps is not None:
        origin = pd.Timestamp(beat_timestamps.iloc[0] if hasattr(beat_timestamps, 'iloc') else beat_timestamps[0])
        timestamps = np.array([
            origin + pd.Timedelta(seconds=i * window_step + window_size)
            for i in range(n_windows)
        ])
    else:
        timestamps = np.array([i * window_step + window_size for i in range(n_windows)], dtype=float)

    return rmssd, timestamps

def moving_average(signal, window_len):
    """
    Smooth a PPG signal using a moving average filter.

    Parameters
    ----------
    signal : array_like
        An array containing the input PPG signal to be filtered.
    window_len : int
        The size of the moving average window, in seconds.

    Returns
    -------
    filtered : array_like
        An array containing the filtered PPG signal.
    """
    kernel = np.ones(window_len) / window_len
    filtered = np.convolve(signal, kernel, mode = 'same')
    return filtered