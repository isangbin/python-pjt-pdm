# 예측 및 조치(제안) 클래스

# SHAP: AI가 왜 그런 판단을 했는지 설명해주는 라이브러리
import shap
import pandas as pd # 데이터 처리를 위해 필요
import numpy as np
from collections import deque # 데이터를 기억하기 위한 큐
import matplotlib.pyplot as plt

class WeldingAdvisor:
    
    # 실시간 데이터를 진단하고, 문제 해결을 위한 구체적인 조치(Action)를 제안하는 클래스
    
    def __init__(self, detector, data_manager):
        # 학습된 모델 가져오기
        self.model = detector.model
        # DataManager에서 계산해둔 정상 기준값(평균, 표준편차)을 가져옴
        self.normal_means = data_manager.normal_means
        self.normal_stds = data_manager.normal_stds 
        # 최근 데이터를 저장할 저장소 (최대 15개까지만 기억)
        # 15개인 이유: 너무 짧으면 노이즈에 민감하고, 너무 길면 반응이 느림
        self.window_size = 15
        self.history = deque(maxlen=self.window_size)
        
        # 각 변수별 정상으로 간주하는 기울기 임계값 (표준편차의 5% 이상 변하면 급변으로 간주)
        # 예: 표준편차가 10인데, 매번 0.5씩 오른다면 20번 뒤엔 10(1시그마)만큼 변하므로 유의미함
        self.slope_thresholds = {col: (sigma * 0.05) for col, sigma in self.normal_stds.items()}

        # 연속 시각화 위한 설정
        plt.ion()   # 대화형모드(interactive mode) 키는 것
        self.fig, self.ax = plt.subplots(figsize=(10, 5)) # 그래프 창 미리 생성

    def monitor_stream(self, new_data_row):
        """
        [핵심 기능] 데이터가 실시간으로 한 줄씩 들어올 때마다 호출되는 함수
        데이터를 저장소에 쌓고, 추세와 위험도를 분석합니다.
        """
        # 1. 히스토리에 데이터 추가 (DataFrame -> dict 변환 후 저장)
        # 실제 현장에서는 센서값이 딕셔너리나 리스트로 옴
        current_values = new_data_row.iloc[0].to_dict()
        self.history.append(current_values)
        
        # 2. 모델 예측 (현재 순간의 위험도)
        prob_defect = self.model.predict_proba(new_data_row)[0][1]
        
        # 3. 분석 및 리포트 작성
        print(f"\n--- [실시간 모니터링] (버퍼: {len(self.history)}/{self.window_size}) ---")
        
        # (1) 현재 순간의 위험도 체크 (기존 로직)
        if prob_defect > 0.6:
            print(f"현재 데이터 불량 위험 높음! (확률: {prob_defect*100:.1f}%)")
            self._analyze_root_cause(new_data_row) # 원인 분석
        
        # (2) 추세(Trend) 분석 - 데이터가 어느 정도(5개 이상) 쌓여야 가능
        elif len(self.history) >= 5:
            trend_warnings = self._check_trends()
            
            if trend_warnings:
                print(f"[WARNING] 불량 전조 증상(추세) 감지됨!")
                for msg in trend_warnings:
                    print(f"   - {msg}")
            else:
                print(f"[SAFE] 현재 상태 양호하며, 특이한 추세 없음.")
        else:
             print(f"[INFO] 추세 분석을 위해 데이터를 수집 중입니다...")

    def _check_trends(self):
        """
        저장된 최근 데이터(history)를 바탕으로 각 변수의 기울기를 계산
        """
        warnings = []
        df_history = pd.DataFrame(list(self.history)) # 큐를 데이터프레임으로 변환
        
        # 시간 축 (0, 1, 2, ... N)
        x = np.arange(len(df_history))
        
        for col in self.normal_means.index:
            y = df_history[col].values
            
            # 1차 함수(직선)로 근사하여 기울기(slope) 계산
            # polyfit(x, y, 1) -> [기울기, 절편] 반환
            slope, _ = np.polyfit(x, y, 1)
            
            threshold = self.slope_thresholds[col]
            current_val = y[-1]
            mean_val = self.normal_means[col]

            # 로직: 기울기가 가파르고(threshold 초과) + 정상 평균에서 멀어지는 방향일 때 경고
            
            # 1) 값이 상승세(+)이고, 현재 값이 평균보다 높은데 더 올라가려 함 (악화)
            if slope > threshold and current_val > mean_val:
                warnings.append(f"'{col}'이(가) 급격히 상승 중 (기울기: +{slope:.2f}). 낮추는 조치 필요.")
            
            # 2) 값이 하락세(-)이고, 현재 값이 평균보다 낮은데 더 내려가려 함 (악화)
            elif slope < -threshold and current_val < mean_val:
                warnings.append(f"'{col}'이(가) 급격히 하락 중 (기울기: {slope:.2f}). 높이는 조치 필요.")
                
        return warnings

    def _analyze_root_cause(self, input_data):
        """(기존과 동일) Z-Score 기반의 상세 원인 분석"""
        print("   >> 상세 조치 가이드:")
        for name in input_data.columns:
            val = input_data[name].values[0]
            mean = self.normal_means[name]
            std = self.normal_stds[name]
            z = (val - mean) / std
            
            if abs(z) > 1.5:
                direction = "낮춰야" if z > 0 else "높여야"
                print(f"      * {name}: {val:.1f} (정상 {mean:.1f} 대비 {abs(val-mean):.1f} 차이) -> **{direction}** 함")
    
    # WeldingAdvisor 클래스 내부에 추가
    def plot_live_trend(self, feature_name):
        """특정 변수의 최근 흐름과 미래 예측 점선을 시각화합니다."""
        if len(self.history) < 5: return # 데이터가 충분치 않으면 생략
        
        df_hist = pd.DataFrame(list(self.history))
        y = df_hist[feature_name].values
        x = np.arange(len(y))
        
        # 1차 회귀선(기울기) 계산
        slope, intercept = np.polyfit(x, y, 1)
        
        # 미래 5포인트 예측값 계산
        future_x = np.arange(len(y), len(y) + 5)
        future_y = slope * future_x + intercept
        
        # 기존 그래프 지우고 새로그리기
        self.ax.clear()
        # 현재까지의 실선 그래프
        self.ax.plot(x, y, label='Actual Data', marker='o', color='blue')
        # 미래 예측 점선 그래프
        self.ax.plot(future_x, future_y, label='Forecast Trend', linestyle='--', color='red', marker='x')
        
        # 정상 범위(평균 +- 2*표준편차) 표시
        mean = self.normal_means[feature_name]
        std = self.normal_stds[feature_name]
        self.ax.axhline(mean + 2*std, color='orange', linestyle=':', label='Warning Upper Bound')
        self.ax.axhline(mean - 2*std, color='orange', linestyle=':', label='Warning Lower Bound')
        
        self.ax.set_title(f"Live Trend & Forecast: {feature_name} (Continuous)")
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
        
        # plt.show() 대신 화면 갱신만 수행
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1) # 화면 업데이트를 위한 짧은 휴식