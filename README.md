## ADVANCED LDA
### LDA
![image](https://github.com/user-attachments/assets/35f4dae8-8641-4da8-bd49-481cba1bdf8c)
![image](https://github.com/user-attachments/assets/b8c4470b-e959-43c9-957b-67313d5f3b9b)
![image](https://github.com/user-attachments/assets/11173a11-dc4e-44ec-ba6d-ae7bcd322400)


### avx
>작은 사이즈의 연산들을 각각처리 하는 것이 아니라 한번에 커다란 연산들로 처리하는것
```
1 .코드가 사용하는 avx256 버전에서 double형이 256bit들어갈 수 있는 최대word 수(256/(doble(8-byte)*1byte(8bit))=4)를 multiplication크기로 설정
2. 행렬 B[j,k) value를 multiplication크기만큼 벡터 복제 저장한 후<-해당 코드에서 broadcast가 수행
3. 행렬 &A[k,i]주소+multiplication크기(4)만큼 데이터를 가져와<-해당 코드에서 load가 수행
4. 둘을 곱한 후 &C[j,i]주소+multiplication크기(4)에 넣는 것 ,_해당 코드에서 fmadd,store가 수행
```
```
for (int i=si;i<si+block;i+=loop*16)
avx512에서 integer크기를 고려 (512/(integer(4byte)*8it)=16 주소+16만큼 데이터를 가져와 멀티계산 이때 i는 16만큼 이동해야 데이터 중복이 사라짐.

            {
               for (int j=sj;j<sj+block;j++)
               {
                 __m512i c[loop];
                 for (int x=0;x<loop;x++)
                   c[x]=_mm512_loadu_epi32(&C[j][i+x*16]);
int32형자료를 총 16개를 가져와 _m512i형태의 벡터로 저장. 이과정의 경우 저장하려는 C에 원래 값이 있을 수도 있고 C는 반복해서 저장 데이터를 쌓아 가는 것이기 때문에 따로 변수(c[x])를 두어 계산에 용이성을 제공 .

                 for (int k=sk;k<sk+block;k++)
                 {
                   __m512i b0=_mm512_set1_epi32(B[j][k]);
. 행렬 B[j,k] value를 multiplication크기(16)만큼 벡터 복제하는 과정. 기존에 broadcast가 해당 역할을 수행하던것을 set1으로 대치.

                   for (int x=0;x<loop;x++)
                   {
                   c[x]=_mm512_add_epi32(c[x],_mm512_mullo_epi32(_mm512_loadu_epi32(&A[k][i+x*16]),b0));
.벡터 계산 위의 fmadd함수가 수행하던 것을 두개의 과정(add,mullo)으로 나누어 A의 자료를 순서대로 16개 가져와 B의 복사된 벡터와 곱연산을 한 후 위에서 미리 준비한 c[x]에 더하여 저장.

                   }
                 }
                 for (int x=0;x<loop;x++)
                   _mm512_storeu_epi32(&C[j][i+x*16],c[x]);
.c[x]를 C에 맞는 위치[&C[j][i]]+16만큼의 주소에 맞게 저장

               }
            }
```
### Blocking
>하나의 matrix를 그대로 계산하는 것인 아닌 여러 개의 block으로 나누어 한 번 데이터를 캐쉬에 불러왔을 때 재사용 확률(miss rate낮춤)을 높인다. 
```
int block=128;
    for (int sj = 0;sj < size; sj+=block)
      {
        for (int si = 0; si < size; si+=block)
        {
          for (int sk = 0; sk < size; sk+=block)
          {
for (int i=si;i<si+block;i+=loop*16)
            {
               for (int j=sj;j<sj+block;j++)
               {
                
                 {
                   for (int x=0;x<loop;x++)
                   {
                   }
               
               }
              }
             }
```


### Loop unrolling
>Loop를 parallelism하게 expose하여 루프 컨트롤 오버헤드를 줄이는 것.
```
int loop=2;

            for (int i=si;i<si+block;i+=loop*16)
하나의 loop를 여러 loop로 나눈 만큼 나눈 loop만큼 곱하여 하나의 loop크기만큼 움직일 수 있게 한다.

            {
               for (int j=sj;j<sj+block;j++)
               {
                 __m512i c[loop];
                 for (int x=0;x<loop;x++)
                   c[x]=_mm512_loadu_epi32(&C[j][i+x*16]);
 나눈 loop만큼 원래 c[x]에 C의 matrix를 한번에 저장하는 것을 loop만큼 나누어 오버헤드를 줄인다(원래 32개의 데이터를 한번에 가져와왔다면 loop로 나누어 계산).

                 for (int k=sk;k<sk+block;k++)
                 {
                   __m512i b0=_mm512_set1_epi32(B[j][k]);
                   for (int x=0;x<loop;x++)
                   {
                   c[x]=_mm512_add_epi32(c[x],_mm512_mullo_epi32(_mm512_loadu_epi32(&A[k][i+x*16]),b0));
                   }
                 }
                 for (int x=0;x<loop;x++)
                   _mm512_storeu_epi32(&C[j][i+x*16],c[x]);
 마찬 가지로 나눈 loop만큼 원래 matrix 계산과 저장 과정을 loop만큼 나누어 오버헤드를 줄인다.

               }
            }
```

## 결과
![image](https://github.com/user-attachments/assets/bb1f2017-ef16-4cdb-b92c-959a408cffc2)


### Branch predictor 
![image](https://github.com/user-attachments/assets/cd173ffc-a28d-4b5c-adcc-5fb84415e3cb)
