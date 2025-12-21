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

# 학습/검증용 데이터 분할
from sklearn.model_selection import train_test_split

# PyTorch 모델 구축 및 연산
import torch
import torch.nn as nn
import torch.nn.functional as F

# 최적화 및 학습률 조정
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 데이터셋 및 데이터로더 처리
from torch.utils.data import Dataset, DataLoader

# 분류 모델 평가 지표 계산
from torchmetrics.classification import BinaryF1Score


# 데이터 및 라벨 경로 설정
DATA_PATH = "./20fps_merged_data.csv"
LABEL_PATH = "./Label.csv"


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

def main():
    # 데이터 전처리
    data_df = preprocess(DATA_PATH)
    
    # 라벨 CSV 파일 -> DataFrame 변환
    label_df = pd.read_csv(LABEL_PATH)


if __name__ == "__main__":
    main()