# 우리가 만든 파일(모듈)들에서 클래스를 가져옵니다.
from src.loader import WeldingDataManager
from src.model import WeldingFaultDetector
from src.advisor import WeldingAdvisor
# import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # 1. 데이터 준비 단계
    print("--- 1. 데이터 준비 ---")
    dm = WeldingDataManager() # 데이터 매니저 생성
    # 데이터 로드 (파일 경로 지정)
    dm.load_data('data/normal_data.csv', 'data/outlier_data.csv') 
    # 데이터를 6:2:2로 나눔
    dm.split_data()

    # 2. 모델 학습 단계
    print("\n--- 2. 모델 학습 및 평가 ---")
    detector = WeldingFaultDetector() # 모델 감지기 생성
    detector.train(dm)   # 학습 시작
    detector.evaluate(dm) # 최종 점검
    
    # (선택사항) 변수 중요도 그래프를 보고 싶으면 아래 주석 해제
    # detector.show_feature_importance()

    """학습 직후 시각화 함수 부분"""
    # 모델 신뢰도 확인
    detector.plot_confusion_matrix(dm)
    # 모델이 학습한 정상 기준이 뭔지 확인 (변수별 분포 범위)
    # 이를 통해 각 변수(센서값)가 어느 정도면 정상인지 시각적으로 파악
    detector.plot_normal_ranges(dm)


    # 어드바이저 생성
    advisor = WeldingAdvisor(detector, dm)
    # 3. 시뮬레이션 시나리오 생성
    print("\n================================================")
    print(" 🚀 [시나리오] 용접기 전압(DV_R)이 서서히 과열되는 상황")
    print("================================================")
    
    # (1) 정상 데이터 하나를 가져옵니다.
    start_idx = dm.y_test[dm.y_test == 0].index[0]
    base_data = dm.X_test.loc[[start_idx]].copy()
    
    # (2) 15단계에 걸쳐서 DV_R(직류전압) 값을 야금야금 올립니다.
    # 처음엔 정상범위 -> 점점 상승 -> 나중엔 모델이 불량으로 판정
    print(f"초기 상태: {base_data.iloc[0].to_dict()}")
    
    for step in range(1, 16):
        # 데이터 복제
        current_data = base_data.copy()
        
        # 인위적인 노이즈(진동) + 지속적인 상승 트렌드(step * 1.5) 추가
        noise = np.random.normal(0, 0.5) 
        drift = step * 2.0 # 단계마다 전압이 2.0씩 오름 (꽤 빠른 속도)
        
        current_data['DV_R'] = current_data['DV_R'] + drift + noise
        
        # 모니터링 시스템에 데이터 주입
        print(f"\n[Step {step}] 입력 DV_R: {current_data['DV_R'].values[0]:.2f}")
        # 실시간 모니터링 및 추세 분석
        advisor.monitor_stream(current_data)

        # 실시간 추세 및 미래 예측 그래프 시각화 (advisor.py의 plot_live_trend 호출)
        # 5개 이상의 데이터가 쌓인 시점(Step 5)부터 그래프를 출력합니다.
        if step >= 5:
            advisor.plot_live_trend('DV_R') 
        
        # 사람이 로그를 읽을 수 있게 약간의 딜레이
        time.sleep(0.5)
    print("\n--- 시뮬레이션 종료 ---")
    plt.ioff()  # 대화형 모드 끄기
    plt.show()  # 창 유지
    '''
    랜덤 뽑아서 양불판정 (기능보류)

    # 3. 실시간 시뮬레이션 단계
    print("\n--- 3. 실시간 예지보전 시뮬레이션 ---")
    # 어드바이저 생성 (학습된 모델과 데이터 정보를 넘겨줌)
    advisor = WeldingAdvisor(detector, dm)

    print(">> 테스트 셋에서 실제 '불량' 데이터를 무작위로 하나 추출합니다.")
    
    # 테스트 데이터(y_test) 중 실제 불량(값이 1)인 것들의 위치(index)를 모두 찾음
    defect_indices = dm.y_test[dm.y_test == 1].index
    
    if len(defect_indices) > 0:
        # [수정] 리스트의 0번째가 아니라, 랜덤으로 하나를 뽑음
        # 이렇게 해야 실행할 때마다 다른 케이스를 분석할 수 있음
        target_idx = random.choice(defect_indices)
        
        print(f"   -> (Random Select) 선택된 데이터 ID: {target_idx}")
        
        # 선택된 인덱스의 입력 데이터(X)를 가져옴
        sample_data = dm.X_test.loc[[target_idx]]
        
        # 어드바이저에게 진단 요청 (새로 바뀐 로직으로 전체 변수를 분석함)
        advisor.diagnose(sample_data)
    else:
        print("테스트 셋에 불량 데이터가 없어서 시뮬레이션을 종료합니다.")
    
    '''

if __name__ == "__main__":
    main()

