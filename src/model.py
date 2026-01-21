# 모델 학습 및 평가 클래스

# XGBoost: 강력한 성능의 트리 기반 머신러닝 라이브러리
import xgboost as xgb
# 그래프 그리기용
import matplotlib.pyplot as plt
# 모델 성능 평가 지표(정밀도, 재현율 등) 출력용
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

class WeldingFaultDetector:
    
    # XGBoost 알고리즘을 사용하여 용접 불량을 탐지하는 모델 클래스
    
    def __init__(self):
        # XGBClassifier 객체 생성 (하이퍼파라미터 설정)
        self.model = xgb.XGBClassifier(
            n_estimators=100,      # 나무(Tree)를 100개 심겠다는 뜻
            max_depth=5,           # 나무의 깊이 (너무 깊으면 과적합, 너무 얕으면 성능 저하)
            learning_rate=0.1,     # 학습 속도 (작을수록 꼼꼼하게, 클수록 빠르게 학습)
            random_state=42,       # 결과 재현을 위한 시드 고정
            early_stopping_rounds=10 # 성능이 10번 연속 좋아지지 않으면 학습을 중단 (과적합 방지)
        )
        
    def train(self, data_manager):
        
        # 준비된 데이터를 받아 모델 학습
        
        print("\n-Model- 학습 시작")
        # fit: 실제 학습을 수행하는 함수
        self.model.fit(
            data_manager.X_train, data_manager.y_train, # 훈련 데이터로 공부
            # 검증 데이터(Val)를 사용하여 학습 중간중간 시험봄
            # 점수가 더 이상 안 오르면 멈추기 위함
            eval_set=[(data_manager.X_val, data_manager.y_val)],
            verbose=False # 학습 과정을 일일이 출력하지 않음
        )
        print("-Model- 학습 완료")

    def evaluate(self, data_manager):
        
        # 학습에 쓰지 않은 테스트 데이터로 최종 성능을 평가
        
        # predict: 문제(X_test)만 보고 정답을 맞혀보게 함
        y_pred = self.model.predict(data_manager.X_test)
        
        print("\n[Test Set 성능 평가]")
        # 실제 정답(y_test)과 예측값(y_pred)을 비교하여 점수표 출력
        print(classification_report(data_manager.y_test, y_pred, target_names=['Normal', 'Defect']))

    # 폐기
    def show_feature_importance(self):
        
        # 어떤 변수(전압, 전류 등)가 불량 판정에 가장 큰 영향을 줬는지 그래프로 보여줌
        
        plt.figure(figsize=(10, 5))
        # importance_type='gain': 해당 변수가 데이터를 얼마나 잘 구분해냈는지(이득) 기준
        xgb.plot_importance(self.model, importance_type='gain', show_values=False)
        plt.title('Feature Importance')
        plt.tight_layout() # 그래프 여백 자동 조정
        plt.show()

    def plot_confusion_matrix(self, data_manager):
        # 테스트 데이터를 활용해 모델의 분류 성능을 히트맵으로 시각화
        y_pred = self.model.predict(data_manager.X_test)
        cm = confusion_matrix(data_manager.y_test, y_pred)
        
        plt.figure(figsize=(7, 5))
        # 정규화된 수치로 히트맵 작성 (annnot=True: 숫자 표시)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal(0)', 'Abnormal(1)'], 
                    yticklabels=['Normal(0)', 'Abnormal(1)'])
        plt.title('Confusion Matrix: Model Performance')
        plt.ylabel('Actual (True)')
        plt.xlabel('Predicted')
        plt.show()

    def plot_normal_ranges(self, data_manager):
        # 학습된 변수별 정상 범위 시각화
        # 데이터 매니저에서 계산된 평균과 표준편차를 가져옴
        means = data_manager.normal_means
        stds = data_manager.normal_stds
        features = data_manager.features

        plt.figure(figsize=(12, 6))
        # 막대 그래프는 평균값을, 에러바(yerr)는 표준편차의 2배(95% 신뢰구간)를 나타냄
        bars = plt.bar(features, means, yerr=stds * 2, capsize=10, color='lightgreen', alpha=0.7, label='Normal Range (Mean ± 2σ)')
        
        plt.title('Step 2: Normal Operation Ranges by Feature', fontsize=14)
        plt.ylabel('Value Scale')
        plt.xlabel('Sensor Features')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 막대 위에 평균값 텍스트 표시
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.legend()
        plt.show()