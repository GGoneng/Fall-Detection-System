# -----------------------------------------------------------------------------------
# 파일명       : Merging_Data.py
# 설명         : 영상별로 추출한 CSV 파일 병합
# 작성자       : 이민하
# 작성일       : 2025-05-03
# 
# 사용 모듈    :
# - pandas       # 데이터프레임 기반 데이터 처리
# - re           # 정규표현식 활용
# - os           # 파일 및 경로 관리
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - 영상별로 따로 추출한 여러 개의 CSV 파일을 하나의 데이터로 변경
# -----------------------------------------------------------------------------------

# 데이터프레임 기반 데이터 처리
import pandas as pd

# 정규표현식 활용
import re

# 파일 및 경로 관리
import os

# CSV 파일들이 있는 폴더 경로
folder_path = './New_Dataset'

# CSV 파일 뒤에 붙은 번호순으로 정렬
csv_files = sorted(
    [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')],
    key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
)
# 모든 CSV 파일을 읽어오되, source_index를 부여
df_list = []

for idx, path in enumerate(csv_files):
    csv_num = int(path.split("_")[3][:4])
    df = pd.read_csv(path)
    df.dropna(inplace = True)
    df['source_index'] = csv_num  # 각 파일에 고유 인덱스 부여
    df_list.append(df)


# 데이터 병합
merged_df = pd.concat(df_list, ignore_index=True)

# 병합된 데이터 CSV 파일로 저장
merged_df.to_csv("20fps_merged_data.csv", index = False)

print(merged_df['source_index'].value_counts().sort_index())
