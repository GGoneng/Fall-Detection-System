# -----------------------------------------------------------------------------------
# 파일명       : Fall_Detection.py
# 설명         : 스켈레톤 데이터를 가지고 낙상 감지 모델 학습
# 작성자       : 이민하
# 작성일       : 2025-12-18
# 
# 사용 모듈    :
# - pandas                           # 데이터프레임 기반 데이터 처리
# - numpy                            # 수치 계산 및 배열 연산
# - os                               # 파일 및 경로 관리
# - sklearn.model_selection          # 학습/검증용 데이터 분할
# - torch, torch.nn, F               # PyTorch 모델 구축 및 연산
# - torch.optim, lr_scheduler        # 최적화 및 학습률 조정
# - torch.utils.data                 # 데이터셋 및 데이터로더 처리
# - torchmetrics.classification      # 분류 모델 평가 지표 계산

# -----------------------------------------------------------------------------------
# >> 주요 기능
# - 스켈레톤 데이터를 RNN 모델 종류를 통해 낙상인지 아닌지 시계열 분석
# - 시계열 모델 간 성능 비교 분석
# -----------------------------------------------------------------------------------


# 데이터프레임 기반 데이터 처리
import pandas as pd

# 수치 계산 및 배열 연산
import numpy as np

# 파일 및 경로 관리
import os

# 난수 생성 및 랜덤 샘플링
import random

# 학습/검증용 데이터 분할
from sklearn.model_selection import train_test_split

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


# 데이터 및 라벨 경로 설정
DATA_PATH = "./20fps_merged_data.csv"
LABEL_PATH = "./Label.csv"

# 변수 설정
BATCH_SIZE = 64



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
                 model_type: str, batch_first: bool=True) -> None:
        super().__init__()
    
        # LSTM 모델
        if model_type == "lstm":
            self.model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=batch_first
            )
        
        elif model_type == "gru":
            self.model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=batch_first
            )     

        self.batch_first = batch_first

        # 출력층
        # 양방향 (시퀀스 데이터에서 더 많은 정보 추출 가능)
        if bidirectional:
            self.output = nn.Linear(hidden_size * 2, 1)
        else:
            self.output = nn.Linear(hidden_size, 1)

    def forward_train(self, inputs, length):
        packed = pack_padded_sequence(
            inputs, length,
            batch_first=self.batch_first,
            enforce_sorted=False
        )

        packed_out, h = self.model(packed)
        out = h[-1]

        return out
    
    def forward_rasp(self, inputs, length):
        output, _ = self.model(inputs)

        out = output[:, length.item() - 1, :]

        return out
        
    def forward(self, inputs, length):
        output, _ = self.model(inputs)
        # 마지막 hidden state 선택
        output = output[:, length - 1, :]
        result = self.output(output)
        
        return result


# 실험 조건 고정 함수
def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 데이터 전처리 함수
def preprocess(data_path: str, frame: int=2600) -> pd.DataFrame:
    # 데이터 CSV 파일 -> DataFrame 변환
    df = pd.read_csv(data_path)

    # 데이터 형태 맞추기
    val = df["source_index"].value_counts().values
    idx = df["source_index"].value_counts().index

    # 13개의 관절을 20fps로 10초짜리 영상에서 추출하므로 13 * 20 * 10 = 2600
    for i, num in enumerate(val):
        if num > frame:
            df[df["source_index"] == idx[i]] = df[df["source_index"] == idx[i]][:frame]

    print(df["source_index"].value_counts())

    # 결측치 제거 (아예 추출이 안된 데이터 제거)
    df.dropna(inplace = True)

    # 정렬
    df = df.sort_values(by=["source_index", "frame"])

    # 좌표(x, y, z) 별로 pivot
    df_pivot = df.pivot(index=["frame", "source_index"], columns="landmark_id", values=["x", "y", "z"])

    # 다중 인덱스 컬럼 -> 단일 문자열 변환 (예: ('x', 11) -> 'x_11')
    df_pivot.columns = [f"{coord}_{int(lid)}" for coord, lid in df_pivot.columns]

    # 인덱스 복구 및 정렬
    df_pivot.reset_index(inplace=True)
    df_pivot = df_pivot.sort_values(by=["source_index", "frame"]).reset_index(drop=True)

    # 랜드마크별 x, y, z 좌표 차이를 구하기 위한 열 선택
    coord_cols = [c for c in df_pivot.columns if c.startswith(('x_', 'y_', 'z_'))]

    df_delta = df_pivot.copy()

    # source_index가 바뀌는 부분에서는 새롭게 좌표 값의 차이를 구하기 위해 source_index 열로 그룹화 후 차이 구하기
    df_delta[coord_cols] = (
        df_pivot
        .groupby('source_index')[coord_cols]
        .diff()
    )

    # 1 프레임은 diff()로 인해 결측치가 되므로 제거
    df_delta = df_delta.dropna().reset_index(drop=True)

    # 결과 CSV파일로 저장
    df_delta.to_csv("20fps_reshaped_data.csv", index=False)

    return df_delta

# DataFrame -> List 전환 함수
def change_type(data_df: pd.DataFrame, label_df: pd.DataFrame) -> Tuple[List, List]:
    x_list = []

    for i in data_df["source_index"].unique():
        x_list.append(np.array(data_df[data_df["source_index"] == i].drop(["frame", "source_index"], axis = 1)))

    y_list = list(label_df["label"])

    return x_list, y_list

# 데이터 로더 처리 함수 (Padding 처리)
def collate_fn(batch):
    feature, target = zip(*batch)

    feature = torch.stack(feature)
    target = torch.stack(target)

    # 각 랜드마크의 좌표가 전부 0일 경우 예외 처리 -> Length에 들어가지 않음
    lengths = torch.stack([
        (length.abs().sum(dim=1) != 0).sum() for length in feature
    ])

    return feature, target, lengths

# 메인 함수
def main():
    # 실험 조건 고정
    set_seed()

    # 데이터 전처리
    data_df = preprocess(DATA_PATH)
    
    # 라벨 CSV 파일 -> DataFrame 변환
    label_df = pd.read_csv(LABEL_PATH)

    # DataFrame -> List 전환
    x_list, y_list = change_type(data_df, label_df)

    # Train: 70%, Val: 15%, Test: 15%
    # Train, Val 데이터 나누기
    X_train, X_val, y_train, y_val = train_test_split(x_list, y_list, 
                                                    test_size=0.3,
                                                    random_state=7,
                                                    stratify=y_list)
    
    # Val, Test 데이터 나누기 
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                    test_size=0.5,
                                                    random_state=7,
                                                    stratify=y_val)
    
    trainDS = DetectionDataset(X_train, y_train)
    valDS = DetectionDataset(X_val, y_val)
    testDS = DetectionDataset(X_test, y_test)

    trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valDL = DataLoader(valDS, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    testDL = DataLoader(testDS, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


if __name__ == "__main__":
    main()