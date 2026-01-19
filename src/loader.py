# 데이터 로딩 및 전처리 클래스

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class WeldingDataManager:
    def __init__(self):
        self.df = None
        self.features = ['DV_R', 'DA_R', 'AV_R', 'AA_R', 'PM_R']
        self.target = 'Target'
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.normal_means = None 

    def load_data(self, normal_data, outlier_data):
        try:
            df_normal = pd.read_csv(normal_data)
            df_abnormal = pd.read_csv(outlier_data)
            
            # 1: 불량, 0: 정상으로 라벨링
            df_normal[self.target] = 0
            df_abnormal[self.target] = 1
            
            self.df = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
            print(f"[Data] 로드 완료: 총 {len(self.df)}건")
            
        except FileNotFoundError:
            print("[Data] 파일을 찾을 수 없어 테스트용 임의 데이터를 생성합니다.")
            self._generate_mock_data()

    def _generate_mock_data(self):
        np.random.seed(42)
        df_norm = pd.DataFrame(np.random.normal(100, 5, (1000, 5)), columns=self.features)
        df_norm[self.target] = 0
        df_abnorm = pd.DataFrame(np.random.normal(110, 10, (200, 5)), columns=self.features)
        df_abnorm[self.target] = 1
        self.df = pd.concat([df_norm, df_abnorm], axis=0).reset_index(drop=True)

    def split_data(self):
        X = self.df[self.features]
        y = self.df[self.target]

        # 6:2:2 분할 로직
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # 정상 데이터의 평균값 계산 (조치 제안 기준)
        self.normal_means = self.X_train[self.y_train == 0].mean()
        print(f"[Data] 분할 완료 -> Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")