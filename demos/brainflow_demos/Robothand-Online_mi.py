# -*- coding: utf-8 -*-
# License: MIT License
"""
SSAVEP Feedback on NeuroScan.

"""

import numpy as np
import time
import numpy as np
import serial
import threading
import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import NeuroScan, Marker,Neuracle
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.utils.model_selection \
    import EnhancedLeaveOneGroupOut
from metabci.brainda.algorithms.decomposition.csp import FBCSP
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import sys
import os
import sys
import os

# from secnet import BaseModel
from metabci.brainda.algorithms.deep_learning.secnet import BaseModel
from metabci.brainda.algorithms.deep_learning.tmransnet import tmransnet
from metabci.brainda.algorithms.deep_learning.cttnet import cttnet

def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


class MaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y



import mne, os, re
import numpy as np


def read_annotations_bdf(annotations):
    pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
    if isinstance(annotations, str):
        with open(annotations, encoding='latin-1') as annot_file:
            triggers = re.findall(pat, annot_file.read())
    else:
        tals = bytearray()
        for chan in annotations:
            this_chan = chan.ravel()
            if this_chan.dtype == np.int32:  # BDF
                this_chan.dtype = np.uint8
                this_chan = this_chan.reshape(-1, 4)
                # Why only keep the first 3 bytes as BDF values
                # are stored with 24 bits (not 32)
                this_chan = this_chan[:, :3].ravel()
                for s in this_chan:
                    tals.extend(s)
            else:
                for s in this_chan:
                    i = int(s)
                    tals.extend(np.uint8([i % 256, i // 256]))

        # use of latin-1 because characters are only encoded for the first 256
        # code points and utf-8 can triggers an "invalid continuation byte"
        # error
        triggers = re.findall(pat, tals.decode('latin-1'))

    events = []
    for ev in triggers:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for description in ev[3].split('\x14')[1:]:
            if description:
                events.append([onset, duration, description])
    return zip(*events) if events else (list(), list(), list())


def readbdfdata(filename, pathname):
    '''
    Parameters
    ----------

    filename: list of str

    pathname: list of str

    Return:
    ----------
    eeg dictionary

    '''

    eeg = dict(data=[], events=[], srate=[], ch_names=[], nchan=[])

    if 'edf' in filename[0]:  ## DSI
        raw = mne.io.read_raw_edf(os.path.join(pathname[0], filename[0]), verbose=False)
        raw.drop_channels("FT7")
        raw.drop_channels("Fpz")
        data, _ = raw[:-1]
        events = mne.find_events(raw)
        ch_names = raw.info['ch_names']
        fs = raw.info['sfreq']
        nchan = raw.info['nchan']
    else:
        # Neuracle
        # read data
        raw = mne.io.read_raw_bdf(os.path.join(pathname[0], 'data.bdf'), preload=False, verbose=False)
        ch_names = raw.info['ch_names']
        data, _ = raw[:len(ch_names)]
        fs = raw.info['sfreq']
        nchan = raw.info['nchan']
        # read events
        try:
            annotationData = mne.io.read_raw_bdf(os.path.join(pathname[0], 'evt.bdf'), verbose=False)
            try:
                tal_data = annotationData._read_segment_file([], [], 0, 0, int(annotationData.n_times), None, None)
                # print('mne version <= 0.20')
            except:
                idx = np.empty(0, int)
                tal_data = annotationData._read_segment_file(np.empty((0, annotationData.n_times)), idx, 0, 0,
                                                             int(annotationData.n_times), np.ones((len(idx), 1)), None)
                # print('mne version > 0.20')
            onset, duration, description = read_annotations_bdf(tal_data[0])
            onset = np.array([i * fs for i in onset], dtype=np.int64)
            duration = np.array([int(i) for i in duration], dtype=np.int64)
            desc = np.array([int(i) for i in description], dtype=np.int64)
            events = np.vstack((onset, duration, desc)).T
        except:
            # print('not found any event')
            events = []

    eeg['data'] = data
    eeg['events'] = events
    eeg['srate'] = fs
    eeg['ch_names'] = ch_names
    eeg['nchan'] = nchan
    return eeg




# 训练模型


def train_model(X, y, srate=1000):
    """使用SecNet的BaseModel训练模型"""
    # 确保输入是正确的形状
    y = np.reshape(y, (-1))
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    print('X_train',X_train.shape)
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 配置模型参数
    configs = {
        'class_num': len(np.unique(y)),  # 分类数量，根据实际标签确定
        'channelNum': 58,  # 通道数量
        'width': 300,  # 时间点数量
        'drop_att': 0.2,  # 注意力机制的dropout概率
        'p': 3  # 其他自定义参数
    }

    # 初始化模型
    model = BaseModel(configs)
    # model = tmransnet()
    # model = cttnet()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)

    return model, device

# 预测标签

def model_predict(X, srate=1000, model=None, device=None):
    """使用训练好的BaseModel进行预测"""
    if model is None:
        raise ValueError("模型未初始化，请先训练模型")

    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        outputs, _ = model(X_tensor)
        _, preds = torch.max(outputs, 1)

    return preds.cpu().numpy()

# 计算离线正确率


def offline_validation(X, y, srate=1000):
    """使用交叉验证评估BaseModel模型性能"""
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)
    kfold_accs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        # 训练模型，获取模型和设备
        model, device = train_model(X_train, y_train, srate=srate)
        model.to(device)

        # 调整测试数据形状
        X_test_reshaped = np.reshape(X_test, (-1, X_test.shape[-2], X_test.shape[-1]))

        # 预测时，只传模型和设备到 model_predict
        p_labels = model_predict(X_test_reshaped, srate=srate, model=model, device=device)
        kfold_accs.append(np.mean(p_labels == y_test))

    return np.mean(kfold_accs)


class FeedbackWorker(ProcessWorker):
    def __init__(self,
                 stim_interval,
                 stim_labels,
                 srate,
                 lsl_source_id,
                 timeout,
                 worker_name,
                 left_com='COM4',  # 左手机械手串口
                 right_com='COM5',  # 右手机械手串口
                 baudrate=57600):  # 波特率
        # 初始化父类
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

        # 机械手配置
        self.left_com = left_com
        self.right_com = right_com
        self.baudrate = baudrate
        self.serial_left = None  # 左手串口对象
        self.serial_right = None  # 右手串口对象
        self.estimator = None
        self.device = None
    def pre(self):
        """预处理阶段：初始化模型和串口"""
        # 1. 初始化串口连接
        try:
            self.serial_left = serial.Serial(
                port=self.left_com,
                baudrate=self.baudrate,
                timeout=0.1
            )
            print(f"✅ 已连接左手机械手: {self.left_com}")
        except Exception as e:
            print(f"❌ 左手机械手连接失败: {e}")
            self.serial_left = None

        try:
            self.serial_right = serial.Serial(
                port=self.right_com,
                baudrate=self.baudrate,
                timeout=0.1
            )
            print(f"✅ 已连接右手机械手: {self.right_com}")
        except Exception as e:
            print(f"❌ 右手机械手连接失败: {e}")
            self.serial_right = None

        # 2. 加载数据并训练模型
        filename = ['data.bdf', 'evt.bdf']
        print('加载数据文件...')
        file = readbdfdata(filename, [r"D:\Research\MetaBCI\MetaBCI-master2\MetaBCI-master\my_data"])
        print('数据加载完成')
        sfreq = file['srate']
        print(f'采样率: {sfreq}')
        nchannel = file['nchan']
        print(f'通道数: {nchannel}')

        # 处理标签
        event = file['events']
        labels = np.delete(event, 1, axis=1)  # 删除第二列
        print(f'事件标签: {labels}')

        # 处理数据
        data = file['data']
        print(f'原始数据形状: {data.shape}')
        data = data[1:-5, :]  # 去除无关通道
        print(f'处理后数据形状: {data.shape}')

        # 截取trial
        trial_length = 4000
        gesture_types = labels[:, 1] - 1  # 标签从0开始

        # 提取所有trial
        trials = []
        for idx, (start, hand) in enumerate(labels):
            end = start + trial_length
            if end > data.shape[1]:
                raise ValueError(f"标签 {idx} 超出数据长度")
            trial_data = data[:, start:end]
            trials.append(trial_data)
        trials_array = np.array(trials)
        print(f'trial数据形状: {trials_array.shape}')

        # 预处理数据（滤波、降采样）
        original_sfreq = 1000
        target_sfreq = 250
        filtered_downsampled_data = []
        for trial in trials_array:
            info = mne.create_info(
                ch_names=[f'ch{i}' for i in range(trial.shape[0])],
                sfreq=original_sfreq,
                ch_types=['eeg'] * trial.shape[0]
            )
            raw = mne.io.RawArray(trial, info)
            raw.filter(l_freq=0.5, h_freq=40, fir_design='firwin')
            raw.resample(target_sfreq)
            filtered_downsampled_data.append(raw.get_data())
        filtered_downsampled_data = np.array(filtered_downsampled_data)
        print(f'预处理后数据形状: {filtered_downsampled_data.shape}')

        # # 离线验证
        # acc = offline_validation(filtered_downsampled_data, gesture_types, srate=self.srate)
        # print(f"离线交叉验证准确率: {acc:.2f}")

        # 训练模型
        model, device = train_model(filtered_downsampled_data, gesture_types, srate=self.srate)
        self.estimator = model  # 只保存模型对象
        self.device = device  # 单独保存设备信息

        # 初始化LSL输出
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id
        )
        self.outlet = StreamOutlet(info)
        print('等待LSL连接...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('LSL连接成功')

    def control_robot(self, label):
        """控制机械手运动"""
        # 确保串口已连接
        if label == 0 and self.serial_left:  # 左手
            try:
                # 机械手控制命令（根据实际协议调整）
                self.serial_left.write(b'A3')  # 初始化命令
                time.sleep(1)
                self.serial_left.write(b'G5')  # 运动命令
                time.sleep(4.5)
                self.serial_left.write(b'G5')  # 复位命令
                time.sleep(2)
                print("✅ 左手机械手运动完成")
            except Exception as e:
                print(f"❌ 左手机械手控制失败: {e}")

        elif label == 1 and self.serial_right:  # 右手
            try:
                self.serial_right.write(b'A3')  # 初始化命令
                time.sleep(1)
                self.serial_right.write(b'G5')  # 运动命令
                time.sleep(4.5)
                self.serial_right.write(b'G5')  # 复位命令
                time.sleep(2)
                print("✅ 右手机械手运动完成")
            except Exception as e:
                print(f"❌ 右手机械手控制失败: {e}")

    def consume(self, data):
        """实时处理：预测并控制机械手"""
        data = np.array(data, dtype=np.float64).T  # 转置为 [通道, 时间]
        data = data[1:-6, :]  # 65通道 → 58通道
        try:
            # 模型预测
            from scipy import signal
            original_sfreq = 1000  # 原始采样率（与离线一致）
            target_sfreq = 250  # 目标采样率（与离线一致）

            # 创建MNE信息对象（通道名、采样率、通道类型）
            info = mne.create_info(
                ch_names=[f'ch{i}' for i in range(data.shape[0])],  # 58个通道
                sfreq=original_sfreq,
                ch_types=['eeg'] * data.shape[0]  # 所有通道均为EEG类型
            )

            # 将数据转换为MNE Raw对象（用于滤波和降采样）
            raw = mne.io.RawArray(data, info)


            raw.filter(
                l_freq=0.5,
                h_freq=40,
                fir_design='firwin',  # 与离线相同的FIR设计
                verbose=False  # 关闭冗余输出
            )
            raw.resample(target_sfreq, verbose=False)
            data_processed = raw.get_data()
            data_reshaped = np.expand_dims(data_processed, axis=0)  # 添加样本维度 → (1, 58, 1000)
            data_reshaped = np.expand_dims(data_reshaped, axis=1)  # 添加通道维度 → (1, 1, 58, 1000)
            # print(f"实时数据调整后形状: {data_reshaped.shape}")  # 确认形状正确
            p_labels = model_predict(data_reshaped, srate=self.srate, model=self.estimator, device=self.device)
            predicted_label = int(p_labels[0])
            print(f"🔮 实时预测标签: {predicted_label} (0=左手, 1=右手)")

            # 控制机械手（用线程避免阻塞）
            threading.Thread(
                target=self.control_robot,
                args=(predicted_label,),
                daemon=True
            ).start()

            # LSL发送结果
            lsl_label = [predicted_label + 1]  # 标签从1开始
            max_attempts = 50
            for attempt in range(max_attempts):
                if self.outlet.have_consumers():
                    try:
                        self.outlet.push_sample(lsl_label)
                        print(f"✅ LSL发送成功 (第{attempt + 1}次)")
                        break
                    except Exception as send_error:
                        print(f"❌ LSL发送失败: {send_error}")
                time.sleep(0.01)

        except Exception as e:
            print(f"⚠️ 实时处理错误: {e}")

    def post(self):
        """清理资源"""
        # 关闭串口
        if self.serial_left and self.serial_left.is_open:
            self.serial_left.close()
            print(f"🔌 已关闭左手机械手连接")
        if self.serial_right and self.serial_right.is_open:
            self.serial_right.close()
            print(f"🔌 已关闭右手机械手连接")
        print("🧹 FeedbackWorker 已关闭")



if __name__ == '__main__':
    # 放大器的采样率
    srate = 1000
    stim_interval = [0, 4]
    stim_labels = list(range(1, 3))
    print(stim_labels)
    cnts = 4  # 数据文件数目

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'
    # print('1')
    worker = FeedbackWorker(
        stim_interval=stim_interval,
        stim_labels=stim_labels,
        srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name
    )

    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels)        # 打标签全为1

    ns = Neuracle(
        device_address=('127.0.0.1', 8712),
        srate=srate,
        num_chans=65)  # NeuroScan parameter
    # 与ns建立tcp连接
    # print(ns.device_address)
    ns.connect_tcp()

    # ns开始采集波形数据
    # ns.start_acq()

    # # register worker来实现在线处理
    ns.register_worker(feedback_worker_name, worker, marker)
    print("已注册的worker：", ns._workers.keys())  # 修改为 _workers

    # 开启在线处理进程
    ns.up_worker(feedback_worker_name)
    print(f"worker 进程状态: {'存活' if worker.is_alive() else '已终止'}")  # 新增

    import threading
    import numpy as np

    time.sleep(1)

    # ns开始截取数据线程，并把数据传递数据给处理进程
    ns.start_trans()

    # 任意键关闭处理进程
    input('press any key to close\n')
    # 关闭处理进程
    ns.down_worker('feedback_worker')

    # 等待 1s
    time.sleep(1)


    # ns停止在线截取线程
    ns.stop_trans()

    # ns停止采集波形数据
    ns.stop_acq()

    ns.close_connection()  # 与ns断开连接

    ns.clear()

    # #
