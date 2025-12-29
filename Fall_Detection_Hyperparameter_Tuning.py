# -----------------------------------------------------------------------------------
# 파일명       : Fall_Detection_Hyperparameter_Tuning.py
# 설명         : RNN 계열 낙상 감지 모델 하이퍼파라미터 튜닝
# 작성자       : 이민하
# 작성일       : 2025-12-18
# 
# 사용 모듈    :
# - pandas                           # 데이터프레임 기반 데이터 처리
# - numpy                            # 수치 계산 및 배열 연산
# - os                               # 파일 및 경로 관리
# - random                           # 난수 생성 및 랜덤 샘플링
# - sklearn.model_selection          # 학습/검증용 데이터 분할
# - torch, torch.nn                  # PyTorch 모델 구축 및 연산
# - torch.optim, lr_scheduler        # 최적화 및 학습률 조정
# - torch.utils.data                 # 데이터셋 및 데이터로더 처리
# - torchmetrics.classification      # 분류 모델 평가 지표 계산
# - typing                           # 타입 힌트 객체
#
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - 스켈레톤 데이터를 RNN 모델 종류를 통해 낙상인지 아닌지 시계열 분석
# - K-Fold를 통해 RNN 모델 간 최적의 하이퍼파라미터 튜닝
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def forward_train(self, inputs: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
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
    def forward_rasp(self, inputs: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        # output의 shape (batch, seq_len, hidden_size)
        # output은 각 timestep(seq_len)별 마지막 hidden_state 접근 가능
        output, h = self.model(inputs)

        # 패딩 처리가 되어 있지 않은 seq까지의 hidden_state 추출
        out = output[:, length.item() - 1, :]

        return out
        
    def forward(self, inputs: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
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
def set_seed(seed: int=7) -> None:
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
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature, target = zip(*batch)

    feature = torch.stack(feature)
    target = torch.stack(target)

    # 각 랜드마크의 좌표가 전부 0일 경우 예외 처리 -> Length에 들어가지 않음
    lengths = torch.tensor([
        max((length.abs().sum(dim=1) > 0).sum().item(), 1) for length in feature
    ], dtype=torch.int64)

    return feature, target, lengths

# Validate 함수
def validating(model: RNNModel, valDL: DataLoader, score_fn: BinaryF1Score, 
               loss_fn: nn.BCEWithLogitsLoss, device: str) -> Tuple[float, float]:
    model.eval()
    score_fn.reset()

    loss_total = 0.0

    with torch.no_grad():
        for feature, target, lengths in valDL:
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
            
            # Score 누적
            score_fn.update(preds, target)
    
    # Score 계산
    final_score = score_fn.compute().item()
    # 배치별 Loss 평균 값
    final_loss = loss_total / len(valDL)
            
    return final_score, final_loss

# Train 함수
def training(model: RNNModel, trainDL: DataLoader, valDL: DataLoader, 
             optimizer: optim.AdamW, epoch: int, score_fn: BinaryF1Score, 
             loss_fn: nn.BCEWithLogitsLoss, scheduler: lr_scheduler.ReduceLROnPlateau, 
             device: str, fold: int) -> Tuple[float, float]:

    # 가중치 파일 저장 위치 정의
    SAVE_PATH = "./saved_models"
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Early Stopping을 위한 변수
    BREAK_CNT_SCORE = 0
    LIMIT_VALUE = 30

    # 가중치 저장을 위한 변수
    SAVE_WEIGHT = os.path.join(SAVE_PATH, f"best_model_weights_{fold}.pth")
    
    LOSS_HISTORY = [[], []]
    SCORE_HISTORY = [[], []]

    for count in range(1, epoch + 1):
        model.train()
        score_fn.reset()

        loss_total = 0

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

            # Score 누적
            score_fn.update(preds, target)

            # 이전 gradient 초기화
            optimizer.zero_grad()

            # 역전파로 gradient 계산
            loss.backward()

            # 계산된 gradient로 가중치 업데이트
            optimizer.step()
        
        # Score 누적
        final_score = score_fn.compute().item()

        # Val Loss, Score 계산
        val_score, val_loss = validating(model, valDL, score_fn, loss_fn, device)
        
        LOSS_HISTORY[0].append(loss_total / len(trainDL))
        SCORE_HISTORY[0].append(final_score)

        LOSS_HISTORY[1].append(val_loss)
        SCORE_HISTORY[1].append(val_score)

        # print(f"[{count} / {epoch}]\n - TRAIN LOSS : {LOSS_HISTORY[0][-1]}")
        # print(f"- TRAIN SCORE : {SCORE_HISTORY[0][-1]}")

        # print(f"\n - VAL LOSS : {LOSS_HISTORY[1][-1]}")
        # print(f"- VAL SCORE : {SCORE_HISTORY[1][-1]}")

        # Val Score 결과로 스케줄러 업데이트
        scheduler.step(val_score)

        # Early Stopping 구현
        if len(SCORE_HISTORY[1]) >= 2:
            if SCORE_HISTORY[1][-1] <= SCORE_HISTORY[1][-2]: BREAK_CNT_SCORE += 1

        if len(SCORE_HISTORY[1]) == 1:
            torch.save(model.state_dict(), SAVE_WEIGHT)
        
        else:
            if SCORE_HISTORY[1][-1] > max(SCORE_HISTORY[1][:-1]):
                torch.save(model.state_dict(), SAVE_WEIGHT)

        if BREAK_CNT_SCORE > LIMIT_VALUE:
            print(f"성능 및 손실 개선이 없어서 {count} EPOCH에 학습 중단")
            break
    
    state_dict = torch.load(
        SAVE_WEIGHT,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    best_score, best_loss = validating(model, valDL, score_fn, loss_fn, device)

    print(f"- {fold} Fold best Score: {best_score}\n- {fold} Fold best Loss: {best_loss}")

    # return LOSS_HISTORY, SCORE_HISTORY 
    return best_score, best_loss


# 메인 함수
def main() -> None:

    # 데이터 및 라벨 경로 설정
    DATA_PATH = "./20fps_merged_data.csv"
    LABEL_PATH = "./Label.csv"

    # 파라미터 설정
    BATCH_SIZE = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCH = 100

    input_size = 39
    model_type = "lstm"

    param_grid = {
        "lr": [5e-3, 1e-3, 5e-4, 1e-4],
        "hidden_size": [32, 64, 128, 256],
        "n_layers": [2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "bidirectional": [True, False],
    }    

    # 실험 조건 고정
    set_seed()

    # 데이터 전처리
    data_df = preprocess(DATA_PATH)
    
    # 라벨 CSV 파일 -> DataFrame 변환
    label_df = pd.read_csv(LABEL_PATH)

    # DataFrame -> List 전환
    x_list, y_list = change_type(data_df, label_df)

    # Train: 90%, Test: 10%
    # Train, Test 데이터 나누기
    X, X_test, y, y_test = train_test_split(x_list, y_list, 
                                                    test_size=0.1,
                                                    random_state=7,
                                                    stratify=y_list)
    
    # 전처리 데이터 저장
    with open("dataset.pkl", "wb") as f:
        pickle.dump(
            {
                "X_train": X,
                "y_train": y,
                "X_test": X_test,
                "y_test": y_test
            },
            f
        )

    results = []

    for params in ParameterGrid(param_grid):

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

        fold_loss, fold_score = [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold+1}")
            
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            
            X_val = [X[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]
            
            vstack_train = np.vstack(X_train)

            scaler = StandardScaler()
            scaler.fit(vstack_train)
            
            X_train_scaled = [scaler.transform(seq).astype(np.float32) for seq in X_train]
            X_val_scaled = [scaler.transform(seq).astype(np.float32) for seq in X_val]

            # Dataset 생성
            trainDS = DetectionDataset(X_train_scaled, y_train)
            valDS = DetectionDataset(X_val_scaled, y_val)
    

            # Dataset -> DataLoader 전환
            trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            valDL = DataLoader(valDS, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

            # 모델 객체 생성
            model = RNNModel(input_size=input_size,
                            hidden_size=params["hidden_size"],
                            n_layers=params["n_layers"],
                            dropout=params["dropout"],
                            bidirectional=params["bidirectional"],
                            model_type=model_type).to(DEVICE)
            
            # Loss, Score 객체 생성
            loss_fn = nn.BCEWithLogitsLoss()
            score_fn = BinaryF1Score().to(DEVICE)
            
            # Optimizer 객체 생성
            optimizer = optim.AdamW(model.parameters(), lr=params["lr"])

            # Learning Rate Scheduler 생성
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)

            # 모델 학습 시작
            score, loss = training(model, trainDL, valDL, optimizer,
                                EPOCH, score_fn, loss_fn, scheduler, DEVICE, 
                                fold + 1)

            fold_score.append(score)
            fold_loss.append(loss)

        # 메모리 정리
        del model
        del optimizer

        torch.cuda.empty_cache()
        gc.collect()
        
        # 결과 출력
        mean_score = np.mean(fold_score)
        std_score = np.std(fold_score)

        results.append({
            "params": params,
            "mean_score": mean_score,
            "std_score": std_score
        })

        print(f"\n\n- Params : {params}")
        print(f"- Final Score Mean : {mean_score:.4f}")
        print(f"- Final Score Std : {std_score:.4f}")

    best_result = max(results, key=lambda x: x["mean_score"])

    print("-----------------Best Parameters-----------------")
    print(f"\n\n- Params : {best_result['params']}")
    print(f"Mean Score : {best_result['mean_score']:.4f}")
    print(f"Std Score : {best_result['std_score']:.4f}")

if __name__ == "__main__":
    main()