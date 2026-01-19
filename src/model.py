# 모델 학습 및 평가 클래스

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class WeldingFaultDetector:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            early_stopping_rounds=10
        )
        
    def train(self, data_manager):
        print("\n[Model] 학습 시작...")
        self.model.fit(
            data_manager.X_train, data_manager.y_train,
            eval_set=[(data_manager.X_val, data_manager.y_val)],
            verbose=False
        )
        print("[Model] 학습 완료.")

    def evaluate(self, data_manager):
        y_pred = self.model.predict(data_manager.X_test)
        print("\n[Test Set 성능 평가]")
        print(classification_report(data_manager.y_test, y_pred, target_names=['Normal', 'Defect']))

    def show_feature_importance(self):
        plt.figure(figsize=(10, 5))
        xgb.plot_importance(self.model, importance_type='gain', show_values=False)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()