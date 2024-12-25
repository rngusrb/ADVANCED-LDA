## ADVANCED LDA
### LDA
1. 클래스별 평균과 공분산 행렬 추정
2. 공통 공분산 행렬 추정
   
![image](https://github.com/user-attachments/assets/35f4dae8-8641-4da8-bd49-481cba1bdf8c)
![image](https://github.com/user-attachments/assets/b8c4470b-e959-43c9-957b-67313d5f3b9b)
![image](https://github.com/user-attachments/assets/11173a11-dc4e-44ec-ba6d-ae7bcd322400)

3.	판별 함수 생성

![image](https://github.com/user-attachments/assets/ced5fdf4-e720-41cc-af51-6e1b30de9c7e

4.	클래스 할당

![image](https://github.com/user-attachments/assets/0354bdb3-fc68-4284-9d31-729bfeff8704)


### LDA 개선
>행렬의 정규화
```
class_sc_mat += ((row - mv) / LA.norm(row - mv)).dot(((row - mv) / LA.norm(row - mv)).T)
```
```
S_B += n * ((mean_vec - overall_mean) / LA.norm(mean_vec - overall_mean)).dot(((mean_vec - overall_mean) / LA.norm(mean_vec - overall_mean)).T)
```
>가중 중앙값 사용
평균 μ_i 를 가중 중앙값 μ ̃로 대체하고자 한다. μ ̃는 다음과 같이 계산한다. i번째 클래스의 샘플 집합 X_i=[x_1^i, x_1^i,….,x_1^i]가 주었을 때 각 변수에 가중치ω_j^i를 주어 이상치에 대처하고 그 값으로 평균을 대체하는 것이다.
```
ω_j^i= 1/(|x_j^i-μ_i^m |+β)
```
```
μ ̃_i= (∑_(j=1)^(n_i)▒〖ω_j^i°x_j^i 〗)/(∑_(j=1)^(n_i)▒ω_j^i )
```
>lda 투영행렬 결정

1.Trace(S_w)>10*Trace(S_b)이면 클래스 내 분산 거리가 너무 분산되었음을 나타내므로, 클래스 내 분산 행렬 S_w^(-1)의 특성 벡터를 구하여 최적의 LDA 투영 행렬로 사용합니다.

2.Trace(S_w)<0.1*Trace(S_b) 클래스 내 분산 거리가 너무 분산되었음을 나타내므로, 클래스 내 분산 행렬 S_b의 특성 벡터를 구하여 최적의 LDA 투영 행렬로 사용합니다.

3. 기타의 경우 원래의 특성벡터를 사용한다.
