# 데이터 로딩 및 전처리 클래스

# 데이터를 다루기 위한 라이브러리 (행렬 연산, 데이터프레임 조작)
import pandas as pd
import numpy as np
# 데이터를 학습용/검증용으로 나누기 위한 함수
from sklearn.model_selection import train_test_split

class WeldingDataManager:
    
    # 데이터를 로드하고 학습용/검증용으로 나누는 클래스
    # 정상의 기준(평균, 표준편차)을 수립
    
    def __init__(self):
        self.df = None
        # 분석할 핵심 변수 5가지
        self.features = ['DV_R', 'DA_R', 'AV_R', 'AA_R', 'PM_R']
        self.target = 'Target'
        
        # 데이터셋 저장소
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        
        # 정상 데이터의 통계 기준값
        self.normal_means = None # 평균 (기준점)
        self.normal_stds = None  # 표준편차 (허용 범위 계산용)

    def load_data(self, normal_path, abnormal_path):
        
        # CSV 파일을 읽어오고 정답(Label)을 붙여 하나로 합침
        
        try:
            df_normal = pd.read_csv(normal_path)
            df_abnormal = pd.read_csv(abnormal_path)
            
            # 0: 정상, 1: 불량으로 라벨링 <- 불량 찾을거니까 직관적으로 편함
            df_normal[self.target] = 0
            df_abnormal[self.target] = 1
            
            self.df = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
            print(f"-Loader- 로드 완료: 총 {len(self.df)}건")
            
        except FileNotFoundError:
            print("-Loader- 파일을 찾을 수 없습니다.")
            # self._generate_mock_data()

    # def _generate_mock_data(self):
    #     np.random.seed(42)
    #     # 정상 데이터: 평균 100, 표준편차 5
    #     df_norm = pd.DataFrame(np.random.normal(100, 5, (1000, 5)), columns=self.features)
    #     df_norm[self.target] = 0
    #     # 불량 데이터: 평균 110, 표준편차 10 (더 많이 흔들림)
    #     df_abnorm = pd.DataFrame(np.random.normal(110, 10, (200, 5)), columns=self.features)
    #     df_abnorm[self.target] = 1
    #     self.df = pd.concat([df_norm, df_abnorm], axis=0).reset_index(drop=True)

    def split_data(self):
        
        # 데이터를 6:2:2로 나누고, 학습용 데이터 중 정상인 것들의 통계 정보를 계산함
        
        X = self.df[self.features]
        y = self.df[self.target]

        # 1차 분할: 전체의 20%를 테스트셋으로 떼어둠
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # 2차 분할: 나머지를 다시 훈련용과 검증용으로 나눔 (최종적으로 6:2:2 비율)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
    
        # 학습 데이터 중 정상(0)인 데이터만 필터링
        normal_data = self.X_train[self.y_train == 0]
        
        # 정상 데이터의 평균을 계산 (기준값)
        self.normal_means = normal_data.mean()
        
        # 정상 데이터의 표준편차를 계산 (변동 폭)
        # 이 값이 있어야 현재 데이터가 평소보다 얼마나 심하게 벗어났는지 알 수 있음
        self.normal_stds = normal_data.std()
        
        print(f"-loader- 분할 및 통계 산출 완료 -> Train: {len(self.X_train)}건")