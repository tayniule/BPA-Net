import os
import scipy.io
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from scipy.signal import resample
import json
import mne.filter

def load_arousal_valence_labels(csv_file):
    """加载Arousal和Valence评分数据"""
    sub_stimulus_labels = {}
    for subject_folder in sorted(os.listdir(csv_file)):
        sub_stimulus_labels[subject_folder] = {}
        subject_path = os.path.join(csv_file, subject_folder)

        if os.path.isdir(subject_path):
            sub = subject_path.split('/')[-1]
            subject_path = os.path.join(subject_path, sub)
            if os.path.isdir(subject_path):
                label_file = os.path.join(subject_path, 'Arousal_Valence.csv')

                try:
                    df = pd.read_csv(label_file, header=None, usecols=[0, 1, 2])
                    df.columns = ['stimulus_id', 'valence', 'arousal']
                    stimulus_labels = {}
                    for _, row in df.iterrows():

                        stimulus_id = int(row['stimulus_id'])
                        stimulus_labels[stimulus_id] = {
                            'valence': float(row['valence']),
                            'arousal': float(row['arousal']),
                            'valence_label': 1 if float(row['valence']) > 5 else 0,
                            'arousal_label': 1 if float(row['arousal']) > 5 else 0
                        }

                    # print(f"成功加载 {len(stimulus_labels)} 个刺激的评分数据")
                    # return stimulus_labels
                    sub_stimulus_labels[subject_folder] = stimulus_labels.copy()
                    # sub_stimulus_labels[subject_folder].append(stimulus_labels.copy())

                except Exception as e:
                    print(f"加载评分数据时出错: {e}")
                    return {}
    return sub_stimulus_labels


def split_by_stimulus_id(signal_data, stimulus_ids, sampling_rate, data_type):
    """根据刺激ID切分信号数据"""
    segments = {}

    change_points = np.where(np.diff(stimulus_ids) != 0)[0] + 1
    change_points = np.concatenate(([0], change_points, [len(stimulus_ids)]))

    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]

        current_stimulus_id = int(stimulus_ids[start_idx])

        if current_stimulus_id != 0:
            segment = signal_data[:, start_idx:end_idx]

            segment_info = {
                'data': segment,
                'start_index': start_idx,
                'end_index': end_idx,
                'duration': segment.shape[1] / sampling_rate,
                'sampling_rate': sampling_rate
            }

            if current_stimulus_id not in segments:
                segments[current_stimulus_id] = []

            segments[current_stimulus_id].append(segment_info)
            # print(segments.keys())

    return segments


def create_multimodal_windows(eeg_segment, gsr_segment, ppg_segment, stimulus_id,
                              fs_eeg, fs_gsr, fs_ppg, window_size=4, overlap=0):
    """
    创建多模态对齐的窗口

    确保三种模态的时间窗口对齐
    """
    # 计算各模态的窗口参数
    window_size_eeg = int(window_size * fs_eeg)
    window_size_gsr = int(window_size * fs_gsr)
    window_size_ppg = int(window_size * fs_ppg)

    overlap_eeg = int(overlap * fs_eeg)
    overlap_gsr = int(overlap * fs_gsr)
    overlap_ppg = int(overlap * fs_ppg)

    step_eeg = window_size_eeg - overlap_eeg
    step_gsr = window_size_gsr - overlap_gsr
    step_ppg = window_size_ppg - overlap_ppg

    # 获取各模态信号长度
    eeg_length = eeg_segment.shape[1]
    gsr_length = gsr_segment.shape[1]
    ppg_length = ppg_segment.shape[1]

    # 计算最小窗口数（确保所有模态都有足够数据）
    num_windows_eeg = (eeg_length - window_size_eeg) // step_eeg + 1
    num_windows_gsr = (gsr_length - window_size_gsr) // step_gsr + 1
    num_windows_ppg = (ppg_length - window_size_ppg) // step_ppg + 1

    num_windows = min(num_windows_eeg, num_windows_gsr, num_windows_ppg)

    if num_windows <= 0:
        return []

    multimodal_windows = []

    for i in range(num_windows):
        # EEG窗口
        eeg_start = i * step_eeg
        eeg_end = eeg_start + window_size_eeg
        eeg_window = eeg_segment[:, eeg_start:eeg_end]

        # GSR窗口
        gsr_start = i * step_gsr
        gsr_end = gsr_start + window_size_gsr
        gsr_window = gsr_segment[:, gsr_start:gsr_end]

        # PPG窗口
        ppg_start = i * step_ppg
        ppg_end = ppg_start + window_size_ppg
        ppg_window = ppg_segment[:, ppg_start:ppg_end]

        window_info = {
            'eeg_data': eeg_window,
            'gsr_data': gsr_window,
            'ppg_data': ppg_window,
            'window_id': i,
            'stimulus_id': stimulus_id,
            'sampling_rates': {
                'eeg': fs_eeg,
                'gsr': fs_gsr,
                'ppg': fs_ppg
            },
            'window_size_seconds': window_size,
            'overlap_seconds': overlap
        }

        multimodal_windows.append(window_info)

    return multimodal_windows

def data_resample(data,f_old,f_new,mode):
    for key in data.keys():
        trial_data=data[key][0]['data']
        n_channels, n_old_points = trial_data.shape

        if mode=='EEG':
            filter_data=prepare(trial_data,f_old,0.1,75)

        if mode=='PPG':
            filter_data=prepare(trial_data,f_old,0.05,20)
        if mode == 'GSR':
            filter_data = prepare(trial_data, f_old,0.05,1.95)

        n_new_points = int(np.ceil(f_new / f_old * n_old_points))
        new_data = resample(filter_data, n_new_points, axis=-1)
        data[key][0]['data']=new_data
        data[key][0]['sampling_rate']=f_new
    return data


def prepare(x: np.ndarray, sfreq,l_freq,h_freq) -> np.ndarray:
    """
    对 NumPy 数组进行带通滤波和陷波滤波。

    Args:
        x (np.ndarray): EEG 数据数组，形状应为 (n_channels, n_times) 或 (n_times, n_channels)。
                        MNE 默认处理 (..., n_times)。
        sfreq (float): 数据的采样频率（例如 250 Hz）。

    Returns:
        np.ndarray: 经过滤波处理后的数据。
    """
    x = x.astype(np.float64)
    # 1. 带通滤波 (0.1Hz - 75Hz)
    # mne.filter.filter_data 接受 NumPy 数组
    x = mne.filter.filter_data(
        data=x,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design='firwin',  # 使用 MNE 推荐的 FIR 滤波器
        verbose=False
    )
    try:
        # 2. 50Hz 陷波滤波
        notch_freq = 50.0
        x = mne.filter.notch_filter(
            x,
            Fs=sfreq,
            freqs=notch_freq,
            method='fir',  # 使用 MNE 默认的 FIR 陷波
            verbose=False
        )
    except:
        pass

    return x

ch=['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', 'O2', 'F7', 'F8', 'T6', 'T4','1','2']


def process_and_save_multimodal_data(base_folder, csv_file, output_file='multimodal_data.pkl', window_size=4,
                                     overlap=1):
    """处理并保存多模态数据"""
    trials_split = {
        'train': range(1,24),
        'val': range(24, 28),
        'test': range(28, 32),
    }

    dataset = {
        'train': list(),
        'val': list(),
        'test': list(),
    }
    # 加载评分数据
    sub_stimulus_labels = load_arousal_valence_labels(csv_file)
    if not sub_stimulus_labels:
        raise ValueError("无法加载评分数据")



    # 遍历所有受试者文件夹
    for subject_folder in sorted(os.listdir(base_folder)):
        subject_path = os.path.join(base_folder, subject_folder)

        if os.path.isdir(subject_path):
            data_file = os.path.join(subject_path, 'datas.mat')

            if os.path.isfile(data_file):
                print(f"处理受试者: {subject_folder}")

                # 读取.mat文件
                mat_data = scipy.io.loadmat(data_file)

                # 提取各模态数据
                eeg_data = mat_data['eeg_datas']
                gsr_data = mat_data['gsr_datas']
                ppg_data = mat_data['ppg_datas']

                fs_eeg = mat_data['fs_eeg'][0, 0]
                fs_gsr = mat_data['fs_gsr'][0, 0]
                fs_ppg = mat_data['fs_ppg'][0, 0]
                fs_new=200

                # 提取刺激ID
                eeg_stimulus_ids = eeg_data[-2, :]
                gsr_stimulus_ids = gsr_data[-2, :]
                ppg_stimulus_ids = ppg_data[-2, :]

                # 移除刺激ID行
                eeg_signals = eeg_data[:-2, :]
                gsr_signals = gsr_data[:-2, :]
                ppg_signals = ppg_data[:-2, :]

                # 按刺激ID切分各模态数据
                eeg_segments_dict = split_by_stimulus_id(eeg_signals, eeg_stimulus_ids, fs_eeg, 'eeg')
                eeg_segments_dict = data_resample(eeg_segments_dict,fs_eeg,fs_new,'EEG')
                gsr_segments_dict = split_by_stimulus_id(gsr_signals, gsr_stimulus_ids, fs_gsr, 'gsr')
                gsr_segments_dict = data_resample(gsr_segments_dict, fs_gsr, fs_new,'GSR')
                ppg_segments_dict = split_by_stimulus_id(ppg_signals, ppg_stimulus_ids, fs_ppg, 'ppg')
                ppg_segments_dict = data_resample(ppg_segments_dict, fs_ppg, fs_new,'PPG')

                subject_windows = []

                # stimulus_id = sub_stimulus_labels[subject_folder]
                for key in sub_stimulus_labels[subject_folder].keys():
                    # key = list(stimulus_dict.keys())[0]
                    if (key in eeg_segments_dict and
                            key in gsr_segments_dict and
                            key in ppg_segments_dict):

                        eeg_segment = eeg_segments_dict[key][0]['data']
                        gsr_segment = gsr_segments_dict[key][0]['data']
                        ppg_segment = ppg_segments_dict[key][0]['data']

                        # 创建多模态窗口
                        multimodal_windows = create_multimodal_windows(
                            eeg_segment, gsr_segment, ppg_segment, key,
                            fs_new, fs_new, fs_new, window_size, overlap
                        )

                        # 为每个窗口添加标签和受试者信息
                        window_id=0
                        for window in multimodal_windows:
                            window.update({
                                'subject_id': subject_folder,
                                'valence_label': sub_stimulus_labels[subject_folder][key]['valence_label'],
                                'arousal_label': sub_stimulus_labels[subject_folder][key]['arousal_label'],
                                'valence_score': sub_stimulus_labels[subject_folder][key]['valence'],
                                'arousal_score': sub_stimulus_labels[subject_folder][key]['arousal']
                            })
                            sample_key=f'/eds-storage/scw/MER/data/{subject_folder}/{subject_folder}-{key}-{window_id}.pkl'
                            sample=np.vstack([window['eeg_data'],window['gsr_data'],window['ppg_data']])
                            data_dict = {
                                'sample': sample, 'label': [sub_stimulus_labels[subject_folder][key]['valence_label'],sub_stimulus_labels[subject_folder][key]['arousal_label']]
                            }
                            os.makedirs(
                                f'/eds-storage/MER/data/{subject_folder}',
                                exist_ok=True)
                            with open(f'{sample_key}', 'wb') as f:  # 注意是 'wb' (write binary) 模式
                                pickle.dump(data_dict, f)
                            window_id+=1
                            mode=[t for t in dataset if key in trials_split[t]][0]
                            dataset[mode].append(sample_key)


    return dataset


if __name__ == "__main__":
    base_folder = '/data/Running/Emotion/Mixed Emotion Recognition/Aligned_data_001'
    csv_file = '/data/Running/Emotion/Mixed Emotion Recognition/Raw_data'

    dataset = process_and_save_multimodal_data(base_folder, csv_file)

    for mode in dataset.keys():
        new_dataset = {
            "subject_data": dataset[mode],
            "dataset_info": {
                "sampling_rate": 200,
                "ch_names": ch,
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0
            }
        }
        with open(f'./{mode}.json', 'w') as f:
            json.dump(new_dataset, f, indent=2)
