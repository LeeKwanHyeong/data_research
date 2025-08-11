# [Transformer]
![img.png](img.png)

# [TITAN]
![img_1.png](img_1.png)

## MAC(Memory as Context)
Titan의 핵심은 '과거로부터 요약된 문맥 메모리'를 현재 입력과 함께 쓰는 것. 컨텍스트 메모리를 동적으로 업데이트하면, 
분포 이동(프로모션/단종/계절성 변화)에 즉각 반응할 수 있다.

## LMM(Local Memory Matching)
'놀라운(surprising) 변화'를 메모리와의 유사도 기준으로 보강하는 모듈. MAC이 '문맥을 불러오기'라면 LMM은
'그 중 꼭 필요한 부분을 고르는' 역할.

## Test-Time Adaption(TTA/TTM)
학습이 끝난 뒤, 테스트 시점에서 아주 소량의 gradient step으로 최신 관측에 맞춰 모델을 재보정한다.
시계열처럼 시간에 따라 분포가 변하는 문제에 효과적이지만, 데이터 누수 방지와 안정성이 중요.
