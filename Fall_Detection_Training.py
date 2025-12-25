# -----------------------------------------------------------------------------------
# 파일명       : Fall_Detection_Training.py
# 설명         : 스켈레톤 데이터를 가지고 낙상 감지 모델 학습
# 작성자       : 이민하
# 작성일       : 2025-12-23
# 
# 사용 모듈    :
# - pandas                           # 데이터프레임 기반 데이터 처리
# - numpy                            # 수치 계산 및 배열 연산
# - os                               # 파일 및 경로 관리
# - random                           # 난수 생성 및 랜덤 샘플링
# - sklearn.model_selection          # 학습/검증용 데이터 분할
# - torch, torch.nn, F               # PyTorch 모델 구축 및 연산
# - torch.optim, lr_scheduler        # 최적화 및 학습률 조정
# - torch.utils.data                 # 데이터셋 및 데이터로더 처리
# - torchmetrics.classification      # 분류 모델 평가 지표 계산
# - typing                           # 타입 힌트 객체
#
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - 스켈레톤 데이터를 RNN 모델 종류를 통해 낙상인지 아닌지 시계열 분석
# - RNN 모델 간 성능 비교
# - 모델 양자화 및 torchscript 성능 비교
# -----------------------------------------------------------------------------------

# 데이터프레임 기반 데이터 처리
import pandas as pd

# 수치 계산 및 배열 연산
import numpy as np

# 파일 및 경로 관리
import os

# 난수 생성 및 랜덤 샘플링
import random

import pickle

import gc

from sklearn.preprocessing import StandardScaler

# 학습/검증용 데이터 분할
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid

# PyTorch 모델 구축 및 연산
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# 최적화 및 학습률 조정
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 데이터셋 및 데이터로더 처리
from torch.utils.data import Dataset, DataLoader

# 분류 모델 평가 지표 계산
from torchmetrics.classification import BinaryF1Score

# 타입 힌트 객체
from typing import List, Tuple


# 학습용 데이터셋
class DetectionDataset(Dataset):
    def __init__(self, data_list: List, label_list: List) -> None:
        self.data_list = data_list
        self.label_list = label_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        featureTS = torch.tensor(self.data_list[idx], dtype=torch.float32)
        targetTS = torch.tensor([self.label_list[idx]], dtype=torch.float32)

        return featureTS, targetTS
    
# 모델
class RNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, 
                 dropout: float, bidirectional: bool, 
                 model_type: str, batch_first: bool=True, 
                 forward_type: str="train") -> None:
        super().__init__()

        # 입력의 일관성을 위한 소문자 변경
        self.model_type = model_type.lower()
        self.forward_type = forward_type.lower()

        # LSTM 모델
        if self.model_type == "lstm":
            self.model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=batch_first
            )

        # GRU 모델
        elif self.model_type == "gru":
            self.model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=batch_first
            )     
        
        # 모델 타입 예외처리
        else:
            raise ValueError(f"모델 타입 에러 (LSTM or GRU) : {model_type}")

        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # 출력층
        # 양방향 (시퀀스 데이터에서 더 많은 정보 추출 가능)
        if bidirectional:
            self.output = nn.Linear(hidden_size * 2, 1)
        else:
            self.output = nn.Linear(hidden_size, 1)

    # 학습용 forward (Pack 사용)
    def forward_train(self, inputs, length):
        length = length.to("cpu")

        packed = pack_padded_sequence(
            inputs, length,
            batch_first=self.batch_first,
            enforce_sorted=False
        )

        if self.model_type == "lstm":
            # h_n의 shape (n_layers * num_directions, batch, hidden_size)
            # h_n는 각 layer별 hidden_state 접근 가능
            packed_out, (h_n, c_n) = self.model(packed)

        elif self.model_type == "gru":
            packed_out, h_n = self.model(packed)
    
        # Pack으로 인해 output을 바로 사용할 수 없으므로 양방향일 경우 수동 concat
        if self.bidirectional:
            forward_out = h_n[-2] # 마지막 Layer의 Forward hidden_state
            backward_out = h_n[-1] # 마지막 Layer의 Backward hidden_state

            out = torch.cat([forward_out, backward_out], dim=-1)
        
        else:
            out = h_n[-1]

        return out
    
    # 라즈베리파이 추론용 forward (배치 사이즈는 무조건 1)
    def forward_rasp(self, inputs, length):
        # output의 shape (batch, seq_len, hidden_size)
        # output은 각 timestep(seq_len)별 마지막 hidden_state 접근 가능
        output, h = self.model(inputs)

        # 패딩 처리가 되어 있지 않은 seq까지의 hidden_state 추출
        out = output[:, length.item() - 1, :]

        return out
        
    def forward(self, inputs, length):
        if self.forward_type == "train":
            output = self.forward_train(inputs, length)
        elif self.forward_type == "rasp":
            output = self.forward_rasp(inputs, length)
        # forward 타입 예외처리
        else:
            raise ValueError(f"forward 타입 에러 (train or rasp) : {self.forward_type}")

        logit = self.output(output)

        return logit


# 실험 조건 고정 함수
def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 데이터 로더 처리 함수 (Padding 처리)
def collate_fn(batch):
    feature, target = zip(*batch)

    feature = torch.stack(feature)
    target = torch.stack(target)

    # 각 랜드마크의 좌표가 전부 0일 경우 예외 처리 -> Length에 들어가지 않음
    lengths = torch.tensor([
        max((length.abs().sum(dim=1) > 0).sum().item(), 1) for length in feature
    ], dtype=torch.int64)

    return feature, target, lengths

# Train 함수
def training(model, trainDL, optimizer,
             epoch, score_fn, loss_fn, 
             device, model_type):

    # 가중치 파일 저장 위치 정의
    SAVE_PATH = "./saved_models"
    os.makedirs(SAVE_PATH, exist_ok=True)

    # 가중치 저장을 위한 변수
    SAVE_WEIGHT = os.path.join(SAVE_PATH, f"best_model_weights_{model_type}.pth")
    
    LOSS_HISTORY = []
    SCORE_HISTORY = []

    for count in range(1, epoch + 1):
        model.train()
        score_fn.reset()

        loss_total = 0
        score_total = 0

        for feature, target, lengths in trainDL:
            feature = feature.to(device)
            target = target.to(device)
            lengths = lengths

            # 결과 추론
            logit = model(feature, lengths)
            
            # 추론값으로 Loss값 계산
            loss = loss_fn(logit, target)
            loss_total += loss.item()

            # 활성화 함수 + 이진 분류 결과로 변경
            probs = torch.sigmoid(logit)
            preds = (probs > 0.5).int()

            # Score 확인
            score_fn.update(preds, target)

            # 이전 gradient 초기화
            optimizer.zero_grad()

            # 역전파로 gradient 계산
            loss.backward()

            # 계산된 gradient로 가중치 업데이트
            optimizer.step()
        
        final_score = score_fn.compute().item()

        LOSS_HISTORY.append(loss_total / len(trainDL))
        SCORE_HISTORY.append(final_score)

        print(f"[{count} / {epoch}]\n - TRAIN LOSS : {LOSS_HISTORY[-1]}")
        print(f"- TRAIN SCORE : {SCORE_HISTORY[-1]}")

        # 가중치 저장
        if len(SCORE_HISTORY) == 1:
            torch.save(model.state_dict(), SAVE_WEIGHT)
        
        else:
            if SCORE_HISTORY[-1] > max(SCORE_HISTORY[:-1]):
                torch.save(model.state_dict(), SAVE_WEIGHT)

    return LOSS_HISTORY, SCORE_HISTORY 

def testing(model, testDL, score_fn, loss_fn, device):
    model.eval()
    score_fn.reset()

    loss_total = 0

    with torch.no_grad():
        for feature, target, lengths in testDL:
            feature = feature.to(device)
            target = target.to(device)
            lengths = lengths

            logit = model(feature, lengths)

            loss = loss_fn(logit, target)
            loss_total += loss.item()

            probs = torch.sigmoid(logit)
            preds = (probs > 0.5).int()

            score_fn.update(preds, target)

    final_score = score_fn.compute().item()
    final_loss = loss_total / len(testDL)

    return final_score, final_loss

# 메인 함수
def main():

    # 데이터 경로 설정
    DATA_PATH = "./dataset.pkl"

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    # 파라미터 설정
    BATCH_SIZE = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCH = 80

    input_size = 39
    model_type = "lstm"

    bidirectional = True
    dropout = 0.1
    hidden_size = 256
    lr = 1e-5
    n_layers = 3
    
    vstack_train = np.vstack(X_train)

    scaler = StandardScaler()
    scaler.fit(vstack_train)
    
    X_train_scaled = [scaler.transform(seq).astype(np.float32) for seq in X_train]
    X_test_scaled = [scaler.transform(seq).astype(np.float32) for seq in X_test]

    # Dataset 생성
    trainDS = DetectionDataset(X_train_scaled, y_train)
    testDS = DetectionDataset(X_test_scaled, y_test)


    # Dataset -> DataLoader 전환
    trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    testDL = DataLoader(testDS, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 모델 객체 생성
    model = RNNModel(input_size=input_size,
                    hidden_size=hidden_size,
                    n_layers=n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    model_type=model_type).to(DEVICE)
    
    # Loss, Score 객체 생성
    loss_fn = nn.BCEWithLogitsLoss()
    score_fn = BinaryF1Score().to(DEVICE)
    
    # Optimizer 객체 생성
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 모델 학습 시작
    loss, score = training(model, trainDL, optimizer,
                        EPOCH, score_fn, loss_fn, DEVICE, 
                        model_type)

    test_score, test_loss = testing(model, testDL, score_fn, loss_fn, DEVICE)

    print(f"\n\nTest Loss : {test_loss}")
    print(f"Test Score : {test_score}")
