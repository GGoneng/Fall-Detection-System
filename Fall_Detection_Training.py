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


# 실험 조건 고정 함수
def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
