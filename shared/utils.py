import os

def get_all_session_files(raw_data_base_path) -> tuple[list[str], list[str]]:
    """
    Get all session combined files from the raw data base path.
    """
    all_sessions = []
    participants = []
    t_list = os.listdir(raw_data_base_path)
    t_list = [t for t in t_list if os.path.isdir(os.path.join(raw_data_base_path, t))]
    for t in t_list:
        patient_id_list = os.listdir(os.path.join(raw_data_base_path, t))
        patient_id_list = [pid for pid in patient_id_list if os.path.isdir(os.path.join(raw_data_base_path, t, pid))]
        participants.extend(patient_id_list)
        for pid in patient_id_list:
            file_list = os.listdir(os.path.join(raw_data_base_path, t, pid))
            file_list = [f for f in file_list if f.endswith('.csv')]
            for f in file_list:
                all_sessions.append(os.path.join(raw_data_base_path, t, pid, f))
    return all_sessions, participants