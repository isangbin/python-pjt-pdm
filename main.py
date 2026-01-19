# 실행 파일

# 분리한 모듈들을 import 합니다.
from src.loader import WeldingDataManager
from src.model import WeldingFaultDetector
from src.advisor import WeldingAdvisor

def main():
    # 1. 데이터 준비
    # 실제 경로가 없다면 내부적으로 임의 데이터를 생성하도록 되어 있습니다.
    dm = WeldingDataManager()
    dm.load_data('data/normal_data.csv', 'data/outlier_data.csv') 
    dm.split_data()

    # 2. 모델 학습 및 평가
    detector = WeldingFaultDetector()
    detector.train(dm)
    detector.evaluate(dm)
    
    # 필요시 주석 해제하여 중요도 그래프 확인
    detector.show_feature_importance()

    # 3. 실시간 진단 시뮬레이션
    advisor = WeldingAdvisor(detector, dm)

    print("\n--- [Simulation] 불량 데이터 테스트 ---")
    # 테스트 셋 중 실제 불량(Target=1)인 데이터 하나를 뽑아옴
    if len(dm.y_test[dm.y_test == 1]) > 0:
        defect_idx = dm.y_test[dm.y_test == 1].index[0]
        sample_data = dm.X_test.loc[[defect_idx]]
        
        # 진단 실행
        advisor.diagnose(sample_data)
    else:
        print("테스트 셋에 불량 데이터가 없어 시뮬레이션을 생략합니다.")

if __name__ == "__main__":
    main()