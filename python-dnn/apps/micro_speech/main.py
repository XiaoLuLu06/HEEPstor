import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录 (HEEPstor/python-dnn/)
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
# 将项目根目录添加到 Python 搜索路径
sys.path.append(project_root)




# micro_speech_pytorch.py
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchaudio
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from pathlib import Path
import tarfile
import urllib.request
import json

from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking
from scipy.io import wavfile
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

import prepare_data

# 添加 HEEPstorch 路径
sys.path.append("C:\\github\\cgra\\heep\\HEEPstor\\python-dnn\\")
sys.path.append("../../")
import heepstorch as hp
import pprint
#import tensorflow as tf

# 参数设置
SAMPLE_RATE = 16000
DURATION = 1000  # ms
WINDOW_SIZE_MS = 30
STRIDE_MS = 20
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

# 频谱尺寸(也可以将wave波形降采样到8000/4000)
SPECTROGRAM_IMAGE_SIZE = 32
N_MELS = 128

# 训练参数

# 计算频谱图参数
window_size_samples = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
stride_samples = int(SAMPLE_RATE * STRIDE_MS / 1000)
length_samples = int(SAMPLE_RATE * DURATION / 1000)


commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go']
#commands = ['yes', 'no']

# 定义变换流水线
transform = transforms.Compose([
#    transforms.ToTensor(), # 转换为张量
#    transforms.Lambda(lambda x: x.unsqueeze(0)), # 增加一个维度
#    transforms.Normalize(mean=[0.5], std=[0.5]), # 标准化
    transforms.Resize((SPECTROGRAM_IMAGE_SIZE, SPECTROGRAM_IMAGE_SIZE)), # 调整图像大小
#    transforms.RandomHorizontalFlip(), # 随机水平翻转
])

class SpeechCommandsDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_index = {label: i for i, label in enumerate(commands)}
        self.mel_spec = MelSpectrogram(sample_rate=16000, n_mels=N_MELS)
        self.db_transform = AmplitudeToDB()
        self.freq_mask = FrequencyMasking(freq_mask_param=15)
        self.time_mask = TimeMasking(time_mask_param=35)

    def load_waveform(self, path):
        sample_rate, data = wavfile.read(path)
        data = data.astype('float32') / 32768.0
        waveform = torch.tensor(data).unsqueeze(0)
        return waveform, sample_rate
    


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        label = self.labels[index]
        waveform, sample_rate = self.load_waveform(path)
        #print("waveform shape: ", waveform.shape)

        spec_torch = torch.stft(
            input=waveform.squeeze(0),
            n_fft=256,
            hop_length=128,
            win_length=256,
            window=torch.hann_window(256),
            return_complex=True # Returns complex tensor
        )
        #print("spec_torch1 shape: ", spec_torch.shape)
        #print("spec_torch1 dtype: ", spec_torch.dtype)
        spec_torch = torch.abs(spec_torch)
        #print("spec_torch2 shape: ", spec_torch.shape)
        spec_torch = spec_torch.unsqueeze(0)
        #print("spec_torch3 shape: ", spec_torch.shape)

        spec_torch = transform(spec_torch)
        #print("spec_torch2 shape: ", spec_torch.shape)
#        spec_torch = spec_torch.squeeze(0)
#        print("spec_torch4 shape: ", spec_torch.shape)


        #spectrogram = tf.signal.stft(
        #    waveform.squeeze(0).numpy(), frame_length=255, frame_step=128
        #)
        #print("spectrogram1 shape: ", spectrogram.shape)  # (124, 129)
        # 将spectrogram转换成torch.tensor类型
        #spectrogram = np.abs(spectrogram)
        #print("spectrogram2 shape: ", spectrogram.shape)

        mel = self.mel_spec(waveform)
        #print("mel0 shape: ", mel.shape)
        mel = self.db_transform(mel)
        #print("mel1 shape: ", mel.shape)
        mel = self.freq_mask(mel)
        mel = self.time_mask(mel)
        mel = mel.squeeze(0).unsqueeze(0)
        #print("mel20 shape: ", mel.shape)

        # 需要将mel的各个维度尺寸都缩小到1/4
        #print("mel21 shape: ", mel.shape)
        #mel = mel.squeeze(0).unsqueeze(0)
        #mel = torch.nn.functional.interpolate(
        #    mel, scale_factor=1/4, mode='linear', align_corners=False
        #)
        #print("mel22 shape: ", mel.shape)

        # 用stft替代mel
        # mel = spectrogram

        #print("mel2 shape: ", mel.shape)
        if mel.size(-1) < 128:
            mel = torch.nn.functional.pad(mel, (0, 128 - mel.size(-1)))
        mel = mel[:, :, :128]
        #print("mel3 shape: ", mel.shape)

        # 缩放1/4
#        mel = F.interpolate(mel, size=[32])
        #print("mel4 shape: ", mel.shape)
        #mel = F.interpolate(mel, scale_factor=0.25)
        #print("mel5 shape: ", mel.shape)

        # mel形状应该是[1, 128, 128]
        #print("mel4 shape: ", mel.shape)

        label_idx = self.label_to_index[label]
        #return mel, label_idx
        return spec_torch, label_idx

class MicroSpeechDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.labels = ['silence', 'unknown', 'yes', 'no']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # 加载数据文件
        if mode == 'train':
            self.data_file = self.data_dir / 'train_list.txt'
        else:
            self.data_file = self.data_dir / 'test_list.txt'
            
        with open(self.data_file, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_path = self.data_dir / self.file_list[idx]
        
        # 解析文件名获取标签
        label_str = audio_path.parent.name
        if label_str in self.label_to_idx:
            label = self.label_to_idx[label_str]
        else:
            label = self.label_to_idx['unknown']
        
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 确保音频长度正确
        if waveform.shape[1] < length_samples:
            padding = length_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :length_samples]
        
        # 转换为频谱图
        spectrogram = self.waveform_to_spectrogram(waveform.squeeze())
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
            
        return spectrogram, label
    
    def waveform_to_spectrogram(self, waveform):
        # 计算MFCC特征
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=FEATURE_BIN_COUNT,
            melkwargs={
                'n_fft': window_size_samples,
                'hop_length': stride_samples,
                'n_mels': FEATURE_BIN_COUNT
            }
        )
        spectrogram = mfcc_transform(waveform)
        return spectrogram

class CNN(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)  # Input: (1, 128, 128)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (32, 64, 64)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)  # Output: (64, 64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (64, 32, 32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)  # Output: (128, 32, 32)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: (128, 16, 16)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (32, 64, 64)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (64, 32, 32)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (128, 16, 16)
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

from torchinfo import summary

class MS(nn.Module):
    def __init__(self, num_classes=8):
        super(MS, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)  # Input: (1, 128, 128)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (32, 64, 64)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)  # Output: (64, 64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (64, 32, 32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)  # Output: (128, 32, 32)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: (128, 16, 16)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        #summary(self, input_size=(1, 64, 124, 129))

    def forward(self, x):
        #print ("input shape: ", x.shape)

        # resize to 1/4?
        #x = self.pool1(x)  # (1,128,128) -> (1,64,64)
        #print ("resize1 shape: ", x.shape)
        #x = self.pool1(x)  # (1,64,64) -> (1,32,32)
        #print ("resize2 shape: ", x.shape)

        x = self.conv1(x)  # (1,32,32) -> (32,30,30)
        #print ("conv1 shape: ", x.shape)

        x = self.conv2(x)  # (32,30,30) -> (64,28,28)
        #print ("conv2 shape: ", x.shape)

        x = self.pool2(x)  # (64,28,28) -> (64,14,14)
        #print ("max pool2 shape: ", x.shape)

        # dropout  # (64,14,14) -> (64,14,14)
        x = self.dropout(x)
        #print ("dropout shape: ", x.shape)

        # flatten  # 12544
        x = nn.Flatten()(x)
        print ("flatten shape: ", x.shape)

        # dense  # 128
        x = self.fc1(x)  # (64,14,14) -> (256)
        #print ("fc1 shape: ", x.shape)

        # dropout # 128

        # dense # num_classes
        x = self.fc2(x)  # (256) -> (num_classes)
        #print ("fc2 shape: ", x.shape)

        return x


def get_model(device):
    """定义micro_speech模型结构"""

    # 32*32
    # model = nn.Sequential(OrderedDict([
    #     ('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=0)),
    #     ('batchnorm1', nn.BatchNorm2d(32)),
    #     ('relu1', nn.ReLU()),
    #     ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=0)),
    #     ('batchnorm2', nn.BatchNorm2d(64)),
    #     ('relu2', nn.ReLU()),
    #     ('maxpool', nn.MaxPool2d(2)),
    #     ('dropout1', nn.Dropout(0.25)),
    #     ('flatten', nn.Flatten()),
    #     ('fc3', nn.Linear(64 * 14 * 14, 128)),
    #     ('relu3', nn.ReLU()),
    #     ('dropout2', nn.Dropout(0.5)),
    #     ('fc4', nn.Linear(128, len(commands))),  # 4个类别: silence, unknown, yes, no
    # ])).to(device)

    # 参数数量57688个，20次训练精度70%多
    # from:https://stackoverflow.com/questions/73456622/recreating-micro-speech-model-from-scratch
    model = nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(1, 8, kernel_size=3)),  # Conv2D: 8 filters, 1×1, stride=2
        ('relu', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('dense', nn.Linear(8 * 30 * 30, len(commands))),  # Dense: 7200 → 4
    ])).to(device)

    # 16*16
    # model = nn.Sequential(OrderedDict([
    #     ('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=0)),
    #     ('batchnorm1', nn.BatchNorm2d(32)),
    #     ('relu1', nn.ReLU()),
    #     ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=0)),
    #     ('batchnorm2', nn.BatchNorm2d(64)),
    #     ('relu2', nn.ReLU()),
    #     ('maxpool', nn.MaxPool2d(2)),
    #     ('dropout1', nn.Dropout(0.25)),
    #     ('flatten', nn.Flatten()),
    #     ('fc3', nn.Linear(64 * 12 * 12, 128)),
    #     ('relu3', nn.ReLU()),
    #     ('dropout2', nn.Dropout(0.5)),
    #     ('fc4', nn.Linear(128, len(commands))),  # 4个类别: silence, unknown, yes, no
    # ])).to(device)


    #model = CNN(num_classes=len(commands)).to(device)
    #model = MS(num_classes=len(commands)).to(device)

    return model


def get_loaders(data_dir, batch_size=64):
    """获取数据加载器"""
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_files, val_files, test_files, train_labels, val_labels, test_labels = prepare_data.prepare_micro_speech_data()
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

    train_dataset = SpeechCommandsDataset(train_files, train_labels)
    #val_dataset = SpeechCommandsDataset(val_files, val_labels)
    test_dataset = SpeechCommandsDataset(test_files, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # train_dataset = MicroSpeechDataset(data_dir, mode='train', transform=transform)
    # test_dataset = MicroSpeechDataset(data_dir, mode='test', transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #print("input data shape=", data.shape)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader, criterion, device, description=""):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set {description}: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def train_model(model, train_loader, test_loader, criterion, device,
                num_epochs=20, lr=0.001, checkpoint_path='micro_speech_model.pth'):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_accuracy = 0
    
    for epoch in range(1, num_epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        accuracy = test(model, test_loader, criterion, device, f"epoch {epoch}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved with accuracy: {best_accuracy:.2f}%")
    
    return model

def load_or_train_model(train_loader, test_loader, criterion, device,
                        retrain=True, checkpoint_path='micro_speech_model.pth'):
    """加载或训练模型"""
    model = get_model(device)
    print("Model summary:")
    summary(model)

    if os.path.exists(checkpoint_path) and not retrain:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
        print(f"Loaded pre-trained model from {checkpoint_path}")
        return model

    return train_model(model, train_loader, test_loader, criterion, device,
                       checkpoint_path=checkpoint_path)

def get_test_predictions(model, test_loader, device, n_samples: int):
    """
    获取测试集预测结果
    """
    # 获取前n_samples个样本
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[:n_samples]
    labels = labels[:n_samples]

    model.eval()
    with torch.no_grad():
        output = model(images.to(device))
        probs = torch.softmax(output, dim=1)
        predictions = probs.argmax(dim=1)

    # 创建可视化
    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
    if n_samples == 1:
        axes = [axes]

    #label_names = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go']
    #label_names = ['yes', 'no']
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze(), cmap='viridis', aspect='auto')
        ax.set_title(
            f'True: {commands[labels[i].item()]}\n'
            f'Pred: {commands[predictions[i].item()]} '
            f'(prob {probs[i, predictions[i]].item() * 100:.2f}%)'
        )
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # 返回矩阵和预测结果
    return {
        'input_matrix': images.numpy(),
        'expected_output_prob_matrix': probs.cpu().numpy(),
        'expected_predictions': predictions.cpu().numpy().tolist(),
        'true_label_values': labels.numpy().tolist()
    }

def print_test_predictions_forward_pass(test_loader, hp_nn: hp.module.SequentialNetwork, n_samples: int):
    """
       Gets first n_samples from test set and computes predictions and matrices
    """
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[:n_samples]
    labels = labels[:n_samples]

    probs = []
    predictions = []
    images_hp = []

    for i in range(n_samples):
        image = hp.code_generator.flatten_input_to_matrix(images[i].detach().numpy())
        output = hp_nn(image)
        all_probs = torch.softmax(torch.from_numpy(output), 1)
        pred = int(torch.argmax(all_probs).item())

        predictions.append(pred)
        probs.append(all_probs[0, pred])
        images_hp.append(image)

    print(f'Heepstorch forward pass predictions: {predictions}')

    # Create visualization
    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
    if n_samples == 1:
        axes = [axes]

    #classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go']
    #classes = ['yes', 'no']

    for i, ax in enumerate(axes):
        # Reshape for display (single channel)
        img_display = images[i].squeeze()  # Remove channel dimension for grayscale
        ax.imshow(img_display, cmap='gray')
        ax.set_title(
            f'True: {commands[labels[i]]}\nPred: {commands[predictions[i]]} \n(prob {probs[i] * 100:.2f}%)')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # 只需要输出一个样本
    image = hp.code_generator.flatten_input_to_matrix(images[0].detach().numpy())
    output = hp_nn(image)
    all_probs = torch.softmax(torch.from_numpy(output), 1)
    prob_matrix = all_probs.numpy()
    pred = int(torch.argmax(all_probs).item())
    expected_predictions = [pred]
    label = labels.numpy().tolist()

    # 返回矩阵和预测结果
    return {
        'input_matrix': image,
        'expected_output_prob_matrix': prob_matrix,
        'expected_predictions': expected_predictions,
        'true_label_values': label
    }

def download_and_prepare_data():
    """下载并准备micro_speech数据集"""
    data_dir = Path('micro_speech_data')
    data_dir.mkdir(exist_ok=True)
    
    # 这里需要根据实际数据源实现数据下载和预处理
    # 由于原始数据可能需要从TensorFlow格式转换，这里简化处理
    print("Please prepare micro_speech dataset in 'micro_speech_data' directory")
    print("Expected structure:")
    print("micro_speech_data/")
    print("  train_list.txt")
    print("  test_list.txt")
    print("  yes/")
    print("  no/")
    print("  silence/")
    print("  unknown/")
    
    return data_dir

def main(retrain=False, use_gpu_if_available=True):
    """主函数"""
    device = torch.device('cuda' if use_gpu_if_available and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 准备数据
    #data_dir = download_and_prepare_data()
    
    # 获取数据加载器
    train_loader, test_loader = get_loaders("./data/mini_speech_commands/", batch_size=32)
    criterion = torch.nn.CrossEntropyLoss()

    # 加载或训练模型
    model = load_or_train_model(train_loader, test_loader, criterion, device, retrain)

    # 转换为HEEPstorch量化模型
    hp_nn = hp.module.SequentialNetwork.from_torch_sequential(model, SPECTROGRAM_IMAGE_SIZE, SPECTROGRAM_IMAGE_SIZE, 1)
    quantized_torch_model = hp_nn.get_quantized_torch_module()

    # 测试量化模型
    test(quantized_torch_model, test_loader, criterion, device, "quantized")
    test(model, test_loader, criterion, device, "non-quantized")

    # 获取预测结果
    pred_res = get_test_predictions(quantized_torch_model, test_loader, device, 1)
    pprint.pprint(pred_res)

    # 应该就是一个样本的iamge和预测结果，非list类型
    pred_res_hq = print_test_predictions_forward_pass(test_loader, hp_nn, 1)
    pprint.pprint(pred_res_hq)

    # 生成C++代码
    cg = hp.code_generator.CodeGenerator('micro_speech', hp_nn)
    cg.generate_code(append_final_softmax=True, overwrite_existing_generated_files=True)
    cg.generate_example_main('main.cpp', pred_res_hq['input_matrix'], 
                           pred_res_hq['expected_output_prob_matrix'],
                           pred_res_hq['expected_predictions'], 
                           pred_res_hq['true_label_values'],
                           overwrite_existing_generated_files=True)

if __name__ == "__main__":
    main(retrain=False, use_gpu_if_available=True)

