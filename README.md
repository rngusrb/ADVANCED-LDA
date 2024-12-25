## ADVANCED LDA
### LDA
1. 클래스별 평균과 공분산 행렬 추정
2. 공통 공분산 행렬 추정
   
![image](https://github.com/user-attachments/assets/35f4dae8-8641-4da8-bd49-481cba1bdf8c)
![image](https://github.com/user-attachments/assets/b8c4470b-e959-43c9-957b-67313d5f3b9b)
![image](https://github.com/user-attachments/assets/11173a11-dc4e-44ec-ba6d-ae7bcd322400)

판별함수

![image](https://github.com/user-attachments/assets/ced5fdf4-e720-41cc-af51-6e1b30de9c7e)


### LDA 개선
>행렬의 정규화
```
class_sc_mat += ((row - mv) / LA.norm(row - mv)).dot(((row - mv) / LA.norm(row - mv)).T)
```
```
S_B += n * ((mean_vec - overall_mean) / LA.norm(mean_vec - overall_mean)).dot(((mean_vec - overall_mean) / LA.norm(mean_vec - overall_mean)).T)
```
>가중 중앙값 사용

