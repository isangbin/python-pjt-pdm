# 예측 및 조치(제안) 클래스 (business logic)

import shap

class WeldingAdvisor:
    def __init__(self, detector, data_manager):
        # detector에서 학습된 모델 객체만 가져옴
        self.model = detector.model
        # data_manager에서 정상 기준값만 가져옴
        self.normal_means = data_manager.normal_means
        self.explainer = None 

    def diagnose(self, input_data, threshold=0.6):
        # 1. 확률 예측
        prob_defect = self.model.predict_proba(input_data)[0][1]
        
        print(f"\n===== [실시간 용접 진단 리포트] =====")
        print(f"입력 데이터: {input_data.to_dict(orient='records')[0]}")
        print(f"불량 예측 확률: {prob_defect * 100:.2f}%")

        if prob_defect > threshold:
            print(">> [DANGER] 이상 징후 감지! 원인을 분석합니다.")
            self._prescribe_solution(input_data)
        else:
            print(">> [SAFE] 공정이 안정적입니다.")

    def _prescribe_solution(self, input_data):
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
            
        shap_values = self.explainer.shap_values(input_data)
        
        feature_names = input_data.columns
        risk_factors = []
        
        for name, current_val, shap_val in zip(feature_names, input_data.iloc[0], shap_values[0]):
            if shap_val > 0: # 불량 요인인 경우
                normal_val = self.normal_means[name]
                diff = current_val - normal_val
                action = "낮춰야" if diff > 0 else "높여야"
                risk_factors.append((name, shap_val, action, abs(diff), normal_val))
        
        risk_factors.sort(key=lambda x: x[1], reverse=True)
        
        print("\n>> [조치 제안 가이드]")
        if not risk_factors:
            print("  - 복합적인 패턴이 감지되었습니다. 전체 공정 점검 요망.")
            return

        for item in risk_factors[:2]:
            name, shap_v, action, diff, norm_v = item
            print(f"  1. 원인: '{name}' (영향도: {shap_v:.4f})")
            print(f"     - 현상: 정상({norm_v:.1f}) 대비 {diff:.1f} 차이")
            print(f"     - 조치: 값을 **{action}** 합니다.")