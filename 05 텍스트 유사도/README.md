# 텍스트 유사도

- 이전 장에서는 텍스트 분류 문제 중에서 감정 분석 문제를 해결하기 위해 데이터를 전처리하고 여러 가지 모델링을 통해 성능을 측정했다. 그뿐만 아니라 한글과 영어 텍스트를 자연어 처리할 때 어떤 부분이 서로 다른지 알아보기 위해 영어 텍스트 분류와 한글 텍스트 분류로 두 가지로 나눠서 알아봤다. 
- 이번 장에서는 자연어 처리의 또 다른 문제인 텍스트 유사도 문제를 해결해 보겠다. 텍스트 유사도 문제란 두 문장(글)이 있을 때 두 문장 간의 유사도를 측정할 수 있는 모델을 만드는 것이다. 텍스트 유사도에 대해서는 3장에 자세히 나와 있으므로 참고하길 바란다. 
- 이번 장에서도 캐글의 대회 중 하나를 해결해 보려고 한다. "Quora Questions Pairs"라는 문제가 이번 장에서 해결할 문제인데, 쿼라(Quora)는 질문을 하고 다른 사용자들로부터 답변을 받을 수 있는 서비스로서 이 서비스에 올라온 여러 질문들 중에서 어떤 질문이 서로 유사한지 판단하는 모델을 만드는 것이 이번 장의 목표다. 
- 이번에는 영어 텍스트와 한글 텍스트를 모두 다루지는 않을 것이다. 쿼라 영어 데이터를 가지고 모델링하고, 한글 데이터를 통해 텍스트 유사도를 측정하는 실습은 진행하지 않는다. 4장에서 진행했던 한글 데이터 처리를 생각해 보면 한글 데이터에 대해서도 텍스트 유사도를 측정하는 것이 어렵지 않을 것이다. 
- 먼저 이번 장에서 다룰 문제와 해당 데이터에 대해 자세히 알아보자.

## 문제 소개

|||
|---|----|
|데이터 이름|Quora Question Pairs|
|데이터 권한|쿼라 권한을 가지고 있으며 캐글 가입 후 데이터를 내려받으면 문제없다.|
|데이터 출처|https://www.kaggle.com/quora-question-pairs/data|

- 쿼라(Quora)는 앞서 설명했듯이 질문과 답변을 할 수 있는 사이트다. 실제로 딥러닝에 대해 공부할 떄도 쿼라의 질문들을 참고하면서 많은 공부를 할 수 있다. 쿼라의 월 사용자는 대략 1억명 정도 된다. 매일 수많은 질문들이 사이트에 올라올 텐데 이 많은 질문 중에는 분명히 중복된 것들이 포함될 것이다. 따라서 쿼라 입장에서는 중복된 질문들을 잘 찾기만 한다면 이미 잘 작성된 답변들을 사용자들이 참고하게 할 수 있고, 더 좋은 서비스를 제공할 수 있게 된다. 
- 참고로 현재 쿼라에서는 이미 중복에 대한 검사를 하고 있다. 앞서 배운 랜덤 포레스트 모델을 통해 중복 질문들을 찾고 있다. 
- 이번 장의 내용은 이전 장과 비슷하게 진행된다. 우선 데이터에 대해 간단히 알아본 후 데이터를 자세히 분석하고 그 결과를 토대로 데이터 전처리를 할 것이다. 이후에는 전처리된 데이터를 활용해 여러 가지 모델링을 진행하고 모델들을 비교하면서 이번 장을 마무리할 것이다. 먼저 데이터에 대해 알아보자.

## 데이터 분석과 전처리

- 데이터를 가지고 모델링 하기 위해서는 데이터에 대한 분석과 전처리를 진행해야 한다. 데이터 분석을 통해 데이터의 특징을 파악하고 이를 바탕으로 데이터 전처리 작업을 진행한다. 여기서는 주로 문장의 길이와 어휘 빈도 문석을 해서 그 결과를 전처리에 적용하고자 한다. 데이터를 내려 받는 것부터 시작해서 데이터를 분석한 후 전처리하는 순서로 진행할 것이다. 4장에서 다룬 내용과 유사하므로 큰 어려움 없이 진행할 수 있을 것이다. 

### 데이터 불러오기와 분석하기
- 이번에 다룰 문제는 앞서 소개한 것처럼 캐글의 대회 중 하나인 "Quora Question Pairs"다. 먼저 해당 데이터를 내려받는 것부터 시작한다. 캐글 문제의 데이터를 내려받으려면 앞서 2장에서 설명했던 것처럼 직접 대회 페이지에서 다운로드 하거나 캐글 API를 사용해서 다운로드할 수 있다. 직접 다운로드하는 경우 캐글에 로그인해 "Quora Question Pairs" 대회에 들어가 대회 규정에 동의한 후 데이터 탭에서 다운로드하면 되고, API를 통해 다운로드하는 경우 명령행을 열고 다음 명령을 입력해서 내려받으면 된다. 

```
kaggle competitions download -C quora-question-pairs
```

- 캐글에서 다운로드한 파일의 압축을 풀고, 아래 3개의 파일을 `data_in` 폴더로 이동한다.
    - sample_submission.csv.zip
    - test.csv.zip
    - train.csv.zip
- 위 3개의 파일을 `data_in` 폴더로 이동시켰다면, 이제 데이터 분석을 시작하도록 하자. 우선 데이터를 분석하기 위한 패키지를 모두 불러온다. 

```python
import zipfile

DATA_IN_PATH = './data_in/'

file_list = ['train.csv.zip', 'test.csv.zip', 'sample_submission.csv.zip']

for file in file_list:
    zipRef = zipfile.ZipFile(DATA_IN_PATH + file, 'r')
    zipRef.extractall(DATA_IN_PATH)
    zipRef.close()
```

```python
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
```

- 보다시피 넘파이, 판다스, 시각화를 위한 라이브러리인 맷플롯립과 시본을 비롯해 기본 내장 패키지인 `os`를 불러왔다. 모두 4장에서 사용했던 라이브러리이며, 자세한 설명은 2장에 나와있다. 
- 학습 데이터를 불러와서 어떤 형태로 데이터가 구성돼 있는지 확인해 보자. 판다스의 데이터 프레임 형태로 불러온다. 

```python
train_data = pd.read_csv(DATA_IN_PATH + 'train.csv')
train_data.head()
```

![스크린샷 2025-04-01 오후 10 03 14](https://github.com/user-attachments/assets/dd12dbe4-90bb-451e-becb-1c21fd615f64)

- 데이터는 `id`, `qid1`, `question1`, `question2`, `is_duplicate` 열로 구성돼 있고 `id`는 각 행 데이터의 고유한 인덱스 값이다. `qid1`과 `qid2`는 각 질문의 고유한 인덱스 값이고, `question1`과 `question2`는 각 질문의 내용을 담고 있다. `is_duplicate`는 0 또는 1을 값으로 가지는, 0이면 두개의 질문이 중복이 아니고, 1이면 두 개의 질문이 중복이라는 것을 의미한다. 데이터를 좀 더 자세히 확인하고 분석해 보자.
- 이번에 사용할 데이터가 어떤 데이터이고, 크기는 어느 정도 되는지 알아보기 위해 데이터 파일의 이름과 크기를 각각 출력해서 확인해 보자.

```python
print("파일 크기 : ")
for file in os.listdir(DATA_IN_PATH):
    if 'csv' in file and 'zip' not in file:
        print(file.ljust(30) + str(round(os.path.getsize(DATA_IN_PATH + file) / 1000000, 2)) + 'MB')
```

```python
파일 크기 : 
test.csv                      477.59MB
train.csv                     63.4MB
sample_submission.csv         22.35MB
```

- 파일 크기를 불러올 떄도 4장과 마찬가지로 해당 경로에서 각 파일을 확인한 후 파일명에 'csv'가 들어가고 'zip'이 들어가지 않는 파일들만 가져와 해당 파일의 크기를 보여준다. 
- 파일의 크기를 보면 일반적인 데이터의 크기와는 다른 양상을 보여준다. 대부분 훈련 데이터가 평가 데이터보다 크기가 큰데, 이번에 사용할 데이터는 평가 데이터(test.csv)가 훈련 데이터(train.csv) 보다 5배 정도 더 큰 것을 알 수 있다. 평가 데이터가 큰 이유는 쿼라의 경우 질문에 대해 데이터 수가 적다면 각각을 검색을 통해 중복을 찾아내는 편법을 사용할 수 있는 데, 이러한 편법을 방지하기 위해 쿼라에서 직접 컴퓨터가 만든 질문 쌍을 평가 데이터에 임의적으로 추가했기 때문이다. 따라서 평가 데이터가 크지만 실제 질문 데이터는 얼마 되지 않는다. 그리고 캐글의 경우 예측 결과를 제출하면 점수를 받을 수 있는데, 컴퓨터가 만든 질문쌍에 대한 예측은 점수에 포함되지 않는다. 
- 먼저 학습 데이터의 개수를 알아보자. 앞서 불러온 데이터의 길이를 출력하자.

```python
print('전체 학습데이터의 개수: {}'.format(len(train_data)))
```

```
전체 학습데이터의 개수: 404290
```

- 결과를 보면 전체 질문 쌍의 개수는 40만 개다. 판다스는 데이터프레임과 시리즈라는 자료구조를 가지고 있다. 데이터프레임이 행렬 구조라면 시리즈는 인덱스를 가지고 있는 배열이다. 지금 하나의 데이터에 두 개의 질문이 있는 구조인데, 전체 질문(두 개의 질문)을 한번에 분석하기 위해 판다스의 시리즈를 통해 두 개의 질문을 하나로 합친다. 
- 참고로 앞으로 진행할 분석 순서는 질문 중복 분석, 라벨 빈도 분석, 문자 분석, 단어 분석이다. 그림 첫 번째 질문 중복 분석부터 시작한다.

```python
train_set = pd.Series(train_data['question1'].tolist() + train_data['question2'].tolist()).astype(str)
train_set.head()
```

```
0    What is the step by step guide to invest in sh...
1    What is the story of Kohinoor (Koh-i-Noor) Dia...
2    How can I increase the speed of my internet co...
3    Why am I mentally very lonely? How can I solve...
4    Which one dissolve in water quikly sugar, salt...
dtype: object
```

- 각 질문을 리스트로 만든 뒤 하나의 시리즈 데이터 타입으로 만든다. 결과를 보면 위와 같은 구조로 합쳐졌다. 기존 데이터에서 질문 쌍의 수가 40만 개 정도이고 각각 질문이 두 개 이므로 대략 80만 개 정도의 질문이 있다. 
- 이제 질문들의 중복 여부를 확인해 보자. 넘파이를 이용해 중복을 제거한 총 질문의 수와 반복해서 나오는 질문의 수를 확인한다. 

```python
print('교육 데이터의 총 질문 수: {}'.format(len(np.unique(train_set))))
print('반복해서 나타나는 질문의 수: {}'.format(np.sum(train_set.value_counts() > 1)))
```

```
교육 데이터의 총 질문 수: 537361
반복해서 나타나는 질문의 수: 111873
```

- 중복을 제거한 유일한 질문값만 확인하기 위해 넘파이 `unique` 함수를 사용했고, 중복되는 질문의 정확한 개수를 확인하기 위해 2개 이상의 값을 가지는 질문인 `value_counts`가 2 이상인 값의 개수를 모두 더했다. 결과를 보면 80만 개의 데이터에서 53만 개가 유니크 데이터이므로 27만 개가 중복돼 있음을 알 수 있고, 27만 개의 데이터는 11만 개 데이터의 고유한 질문으로 이뤄져 있음을 알 수 있다. 
- 이를 맷플롯립을 통해 시각화해 보자. 그래프의 크기는 너비가 12인치이고, 길이가 5인치이며 히스토그램은 `question` 값들의 개수를 보여주며 y축의 크기 범위를 줄이기 위해 `log` 값으로 크기를 줄인다. `x` 값은 중복 개수이며 `y` 값은 동일한 중복 횟수를 가진 질문의 개수를 의미한다.

```python
# 그래프에 대한 이미지 사이즈 선언
# figsize: (가로, 세로) 형태의 튜플로 입력
plt.figure(figsize=(12, 5))
# 히스토그램 선언
# bins: 히스토그램 값들에 대한 버켓 범위
# range: x축 값의 범위
# alpha: 그래프 색상 투명도
# color: 그래프 색상
# label: 그래프에 대한 라벨
plt.hist(train_set.value_counts(), bins=50, alpha=0.5, color= 'r', label='word')
plt.yscale('log')
# 그래프 제목
plt.title('Log-Histogram of question appearance counts')
# 그래프 x 축 라벨
plt.xlabel('Number of occurrences of question')
# 그래프 y 축 라벨
plt.ylabel('Number of questions')
```

![스크린샷 2025-04-02 오후 9 54 44](https://github.com/user-attachments/assets/19bf7bac-5632-42a0-8735-19a915b3ec95)

- 히스토그램을 살펴보면 우선 중복 횟수가 1인 질문들, 즉 유일한 질문이 가장 많고 대부분의 질문이 중복 횟수가 50번 이하다. 그리고 매우 큰 빈도를 가진 질문은 이상치가 될 것이다. 
- 질문의 중복 분포를 통계치로 수치화해서 다른 방향으로 확인해 보자.

```python
print('중복 최대 개수: {}'.format(np.max(train_set.value_counts())))
print('중복 최소 개수: {}'.format(np.min(train_set.value_counts())))
print('중복 평균 개수: {:.2f}'.format(np.mean(train_set.value_counts())))
print('중복 표준편차: {:.2f}'.format(np.std(train_set.value_counts())))
print('중복 중간길이: {}'.format(np.median(train_set.value_counts())))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('제 1 사분위 중복: {}'.format(np.percentile(train_set.value_counts(), 25)))
print('제 3 사분위 중복: {}'.format(np.percentile(train_set.value_counts(), 75)))
```

```
중복 최대 개수: 161
중복 최소 개수: 1
중복 평균 개수: 1.50
중복 표준편차: 1.91
중복 중간길이: 1.0
제 1 사분위 중복: 1.0
제 3 사분위 중복: 1.0
```

- 중복이 최대로 발생한 개수는 161번이고, 평균으로 보면 문장당 1.5개의 중복을 가지며, 표준편차는 1.9다. 중복이 발생하는 횟수의 평균이 1.5라는 것은 많은 데이터가 최소 1개 이상 중복돼 있음을 의미한다. 즉, 중복이 많다는 뜻이다. 이제 박스 플롯을 통해 중복 횟수와 관련해서 데이터를 직관적으로 이해해 보자.

```python
plt.figure(figsize=(12, 5))
# 박스플롯 생성
# 첫번째 파라메터: 여러 분포에 대한 데이터 리스트를 입력
# labels: 입력한 데이터에 대한 라벨
# showmeans: 평균값을 마크함

plt.boxplot([train_set.value_counts()],
             labels=['counts'],
             showmeans=True)
```

![스크린샷 2025-04-02 오후 9 59 34](https://github.com/user-attachments/assets/b326d103-7091-4306-abc7-3ca9291efb87)

- 위 분포는 어떠한가? 중복 횟수의 이상치(outliers)가 너무 넓고 많이 분포해서 박스 플롯의 다른 값을 확인하기조차 어려운 데이터다. 앞서 확인한 데이터의 평균과 최대, 최소 등을 계산한 값과 박스 플롯의 그림을 비교해보자.
- 위에서는 중복 횟수와 관련해서 데이터를 살펴봤다면 이제는 데이터에 어떤 단어가 포함됐는지 살펴보자. 어떤 단어가 많이 나오는지 확인하기 위해 워드클라우드를 사용한다. 이를 위해 `train_set` 데이터를 사용한다. 

```python
from wordcloud import WordCloud
cloud = WordCloud(width=800, height=600).generate(" ".join(train_set.astype(str)))
plt.figure(figsize=(15, 10))
plt.imshow(cloud)
plt.axis('off')
```

![스크린샷 2025-04-02 오후 10 03 53](https://github.com/user-attachments/assets/41fe669e-6547-4093-b0ad-a1e400973ec9)

- 워드클라우드로 그려진 결과를 확인해 보면 best, way, good, difference 등의 단어들이 질문을 할 때 일반적으로 가장 많이 사용된다는 것을 알 수 있다. 특이한 점은 해당 결과에서 'Donald Trump'가 존재하는 것이다. 'Donald Trump'가 존재하는 이유는 선거 기간 중 학습 데이터를 만들었기 때문이라고 많은 캐글러들은 말하고 있다.
- 이제 질문 텍스트가 아닌 데이터의 라벨인 'is_duplicate'에 대해 알아보자. 라벨의 경우 질문이 중복인 경우 1 값과 중복이 아닌 0 값이 존재했다. 라벨들의 횟수에 대해 그래프를 그려보자.

```python
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)
sns.countplot(train_data['is_duplicate'])
```

![스크린샷 2025-04-02 오후 10 07 42](https://github.com/user-attachments/assets/8d5f7f7e-a0d2-420c-a0f4-8a4ab59b8e82)

- 라벨값의 개수를 확인해 보면 총 40만 개의 데이터에서 중복이 아닌 데이터가 25만 개이고, 중복된 데이터가 15만 개다. 이 상태로 학습한다면 중복이 아닌 데이터 25만 개에 의존도가 높아지면서 데이터가 한쪽 라벨로 편향된다. 이러한 경우 학습이 원활하게 되지 않을 수도 있으므로 최대한 라벨의 개수를 균형 있게 맞춰준 후 진행하는 것이 좋다. 많은 수의 데이터를 줄인 후 학습할 수도 있고, 적은 수의 데이터를 늘린 후 학습할 수도 있다.
- 다음으로 텍스트 데이터의 길이를 분석해보자. 이전 장에서 진행한 것과 동일하게 문자(characters) 단위로 먼저 길이를 분석한 후 단어 단어로 길이를 분석하겠다. 우선 문자 단위로 분석하기 위해 각 데이터의 길이를 담은 변수를 생성한다. 

```python
train_length = train_set.apply(len)
```

- 각 데이터의 길이값을 담은 변수를 사용해 히스토그램을 그려보자.

```python
plt.figure(figsize=(15, 10))
plt.hist(train_length, bins=200, range=[0,200], facecolor='r', density=True, label='train')
plt.title("Normalised histogram of character count in questions", fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)
```

![스크린샷 2025-04-02 오후 10 12 29](https://github.com/user-attachments/assets/f33ab04e-1d40-4ccc-b694-df2bfde5b6dc)

- 데이터의 각 질문의 길이 분포는 15\~150에 대부분 모여 있으며 길이가 150에서 급격하게 줄어드는 것을 볼 때 쿼라의 질문 길이 제한이 150 정도라는 것을 추정해 볼 수 있다. 길이가 150 이상인 데이터는 거의 없기 때문에 해당 데이터 때문에 문제가 되지는 않을 것이다.
- 이제 이 길이값을 사용해 여러 가지 통곗값을 확인해 보자.

```python
print('질문 길이 최대 값: {}'.format(np.max(train_length)))
print('질문 길이 평균 값: {:.2f}'.format(np.mean(train_length)))
print('질문 길이 표준편차: {:.2f}'.format(np.std(train_length)))
print('질문 길이 중간 값: {}'.format(np.median(train_length)))
print('질문 길이 제 1 사분위: {}'.format(np.percentile(train_length, 25)))
print('질문 길이 제 3 사분위: {}'.format(np.percentile(train_length, 75)))
```

```python
질문 길이 최대 값: 1169
질문 길이 평균 값: 59.82
질문 길이 표준편차: 31.96
질문 길이 중간 값: 51.0
질문 길이 제 1 사분위: 39.0
질문 길이 제 3 사분위: 72.0
```

- 통계값을 확인해 보면 우선 평균적으로 길이가 60 정도라는 것을 확인할 수 있다. 그리고 중간값의 경우 51 정도다. 하지만 최댓값을 확인해 보면 1169로서 평균, 중간값에 비해 매우 큰 차이를 보인다. 이런 데이터는 제외하고 학습하는 것이 좋을 것이다. 
- 이제 데이터의 질문 길이값에 대해서도 박스 플롯 그래프를 그려서 확인해 보자.

```python
plt.figure(figsize=(12, 5))

plt.boxplot(train_length,
             labels=['char counts'],
             showmeans=True)
```

![스크린샷 2025-04-02 오후 10 20 19](https://github.com/user-attachments/assets/f572794a-0645-4478-b740-a42ca072f93c)

- 분포를 보면 문자 수의 이상치 데이터가 너무 많이 분포해서 박스 플롯의 다른 값을 확인하기 조차 어려운 상태다. 앞서 확인한 데이터의 최대, 평균, 중간 등을 계산한 결과 박스 플롯을 비교해 보자.
- 이제 문자를 한 단위로 하는 것이 아니라 각 데이터의 단어 개수를 하나의 단위로 사용해 길이값을 분석해 보자. 하나의 단어로 나누는 기준은 단순히 띄어쓰기로 정의한다. 우선 각 데이터에 대해 단어의 개수를 담은 변수를 정의하자. 

```python
train_word_counts = train_set.apply(lambda x:len(x.split(' ')))
```

- 띄어쓰기를 기준으로 나눈 단어의 개수를 담은 변수를 정의했다. 이제 이 값을 사용해 앞에서 했던 것과 동일하게 히스토그램을 그려보자.

```python
plt.figure(figsize=(15, 10))
plt.hist(train_word_counts, bins=50, range=[0, 50], facecolor='r', density=True, label='train')
plt.title('Normalised histogram of word count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Prabability', fontsize=15)
```

![스크린샷 2025-04-02 오후 10 24 30](https://github.com/user-attachments/assets/6ac2c342-6064-4d59-9c5c-80dc800fb936)

- 히스토그램을 보면 대부분 10개 정도의 단어로 구성된 데이터가 가장 많다는 것을 볼 수 있다. 20개 이상의 단어로 구성된 데이터는 매우 적다는 것을 확인할 수 있다. 데이터의 단어 개수에 대해서도 각 통곗값을 확인해 보자.

```python
print('질문 단어 개수 최대 값: {}'.format(np.max(train_word_counts)))
print('질문 단어 개수 평균 값: {:.2f}'.format(np.mean(train_word_counts)))
print('질문 단어 개수 표준편차: {:.2f}'.format(np.std(train_word_counts)))
print('질문 단어 개수 중간 값: {}'.format(np.median(train_word_counts)))
print('질문 단어 개수 제 1 사분위: {}'.format(np.percentile(train_word_counts, 25)))
print('질문 단어 개수 제 3 사분위: {}'.format(np.percentile(train_word_counts, 75)))
print('질문 단어 개수 99 퍼센트: {}'.format(np.percentile(train_word_counts, 99)))
```

```
질문 단어 개수 최대 값: 237
질문 단어 개수 평균 값: 11.06
질문 단어 개수 표준편차: 5.89
질문 단어 개수 중간 값: 10.0
질문 단어 개수 제 1 사분위: 7.0
질문 단어 개수 제 3 사분위: 13.0
질문 단어 개수 99 퍼센트: 31.0
```

- 데이터의 문자 단위 길이를 확인했을 때와 비슷한 양상을 보인다. 우선 평균 개수의 경우 히스토그램에서도 확인했던 것처럼 11개가 단어 개수의 평균이다. 그리고 중간값의 경우 평균보다 1개 적은 10개를 가진다. 문자 길이의 최댓값인 경우 1100 정도의 값을 보인다. 단어 길이는 최대 237개다. 해당 데이터의 경우 지나치게 긴 문자 길이와 단어 개수를 보여준다. 박스 플롯을 통해 데이터 분포를 다시한번 확인하자.

```python
plt.figure(figsize=(12, 5))

plt.boxplot(train_word_counts,
             labels=['counts'],
             showmeans=True)
```

![스크린샷 2025-04-02 오후 10 28 55](https://github.com/user-attachments/assets/a2372d13-d60f-4a28-86f0-d5a4fb7fd6ed)

- 문자 길이에 대한 박스 플롯과 비슷한 모양의 그래프를 보여준다. 쿼라 데이터의 경우 이상치가 넓고 많이 분포돼 있음을 알수 있다. 
- 이제 대부분의 분석이 끝났다. 마지막으로 몇 가지 특정 경우에 대한 비율을 확인해 보자. 특수 문자 중 구두점, 물음표, 마침표가 사용된 비율과 수학 기호가 사용된 비율, 대/소문자의 비율을 확인해 본다.

```python
qmarks = np.mean(train_set.apply(lambda x: '?' in x)) # 물음표가 구두점으로 쓰임
math = np.mean(train_set.apply(lambda x: '[math]' in x)) # []
fullstop = np.mean(train_set.apply(lambda x: '.' in x)) # 마침표
capital_first = np.mean(train_set.apply(lambda x: x[0].isupper())) #  첫번째 대문자
capitals = np.mean(train_set.apply(lambda x: max([y.isupper() for y in x]))) # 대문자가 몇개
numbers = np.mean(train_set.apply(lambda x: max([y.isdigit() for y in x]))) # 숫자가 몇개
                  
print('물음표가있는 질문: {:.2f}%'.format(qmarks * 100))
print('수학 태그가있는 질문: {:.2f}%'.format(math * 100))
print('마침표를 포함한 질문: {:.2f}%'.format(fullstop * 100))
print('첫 글자가 대문자 인 질문: {:.2f}%'.format(capital_first * 100))
print('대문자가있는 질문: {:.2f}%'.format(capitals * 100))
print('숫자가있는 질문: {:.2f}%'.format(numbers * 100))
```

```
물음표가있는 질문: 99.87%
수학 태그가있는 질문: 0.12%
마침표를 포함한 질문: 6.31%
첫 글자가 대문자 인 질문: 99.81%
대문자가있는 질문: 99.95%
숫자가있는 질문: 11.83%
```

- 대문자가 첫 글자인 질문과 물음표를 동반하는 질문이 99% 이상을 차지한다. 전체적으로 질문들이 물음표와 대문자로 된 첫 문자를 가지고 있음을 알 수 있다. 그럼 여기서 생각해볼 부분이 있다. 즉, 모든 질문이 보편적으로 가지고 있는 이 특징의 유지 여부에 대해서인데, 모두가 가지고 있는 보편적인 특징은 여기서는 제거한다.
- 지금까지 데이터 분석을 통해 데이터의 구조와 분포를 확인했다. 질문 데이터의 중복 여부 분포, 즉 라벨의 분포가 크게 차이나서 학습에 편향을 제공하므로 좋지 않은 영향을 줄 수 있다. 따라서 전처리 과정에서 분포를 맞추는 것이 좋다. 그리고 대부분의 질문에 포함된 첫 번째 대문자는 소문자로 통일한다. 물음표 같은 구두점은 삭제하는 식으로 보편적인 특성은 제거함으로써 필요한 부분만 학습하게 하는 이점을 얻을 수 있다.


### 데이터 전처리 

- 앞서 데이터를 분석한 결과를 바탕으로 데이터를 전처리해 보자. 먼저 전처리 과정에서 사용할 라이브러리를 불러온다. 

```python
import pandas as pd
import numpy as np
import re
import json

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
```

- 판다스와 넘파이, re, 텐서플로의 케라스 라이브러리를 사용한다. 데이터를 분석할 때와 마찬가지로 경로를 설정하고 학습 데이터를 불러오자.

```python
DATA_IN_PATH = './data_in/'
FILTERS = "([~.,!?\"':;)(])"

change_filter = re.compile(FILTERS)

train_data = pd.read_csv(DATA_IN_PATH + 'train.csv', encoding='utf-8')
```

- 맨 먼저 진행할 전처리 과정은 앞서 분석 과정에서 확인했던 내용 중 하나인 라벨 개수의 균형을 맞추는 것이다. 앞서 분석 과정에서 확인했듯이 중복이 아닌 데이터의 개수가 더욱 많기 때문에 이 경우에 해당하는 데이터의 개수를 줄인 후 분석을 진행하겠다. 먼저 중복인 경우와 아닌 경우로 데이터를 나눈 후 중복이 아닌 개수가 비슷하도록 데이터의 일부를 다시 뽑는다.

```python
train_pos_data = train_data.loc[train_data['is_duplicate'] == 1]
train_neg_data = train_data.loc[train_data['is_duplicate'] == 0]

class_difference = len(train_neg_data) - len(train_pos_data)
sample_frac = 1 - (class_difference / len(train_neg_data))

train_neg_data = train_neg_data.sample(frac = sample_frac)
```

- 먼저 라벨에 따라 질문이 유사한 경우와 아닌 경우에 대한 데이터셋으로 구분한다. 데이터프레임 객체의 `loc`라는 기능을 활용해 데이터를 추출한다. 이 기능을 사용해 라벨이 1인 경우와 0인 경우를 분리해서 변수를 생성한다. 이제 두 변수의 길이를 맞춰야 한다. 우선 두 변수의 길이의 차이를 계산하고 샘플링하기 위해 적은 데이터(중복 질문)의 개수가 많은 데이터(중복이 아닌 질문)에 대한 비율을 계산한다. 그리고 개수가 많은 데이터에 대해 방금 구한 비율 만큼 샘플링하면 두 데이터 간의 개수가 거의 비슷해진다. 샘플링한 이후 각 데이터의 개수를 확인해 보자.

```python
print("중복 질문 개수: {}".format(len(train_pos_data)))
print("중복이 아닌 질문 개수: {}".format(len(train_neg_data)))
```

```
중복 질문 개수: 149263
중복이 아닌 질문 개수: 149263
```

- 샘플링한 후 데이터의 개수가 동일해졌다. 이제 해당 데이터를 사용하면 균형 있게 학습할 수 있을 것이다. 우선 라벨에 따라 나눠진 데이터를 다시 하나로 합치자.

```python
train_data = pd.concat([train_neg_data, train_pos_data])
```

- 이렇게 비율을 맞춘 데이터를 활용해 데이터 전처리를 진행하자. 앞서 전처리에서 분석한 대로 문장 문자열에 대한 전처리를 먼저 진행한다. 우선 학습 데이터의 질문 쌍을 하나의 질문 리스트로 만들고, 정규 표현식을 사용해 물음표와 마침표 같은 구두점 및 기호를 제거하고 모든 문자를 소문자로 바꾸는 처리를 한다.
- 각 데이터에 있는 두 개의 질문을 각각 리스트 형태로 만든 후 각 리스트에 대해 전처리를 진행해서 두 개의 전처리된 리스트를 만들자.

```python
change_filter = re.compile(FILTERS)

questions1 = [str(s) for s in train_data['question1']]
questions2 = [str(s) for s in train_data['question2']]

filtered_questions1 = list()
filtered_questions2 = list()

for q in questions1:
     filtered_questions1.append(re.sub(change_filter, "", q).lower())
        
for q in questions2:
     filtered_questions2.append(re.sub(change_filter, "", q).lower())
```

- 물음표와 마침표 같은 기호에 대해 정규 표현식으로 전처리하기 위해 `re` 라이브러리를 활용한다. 먼저 정규 표현식을 사용할 패턴 객체를 만들어야 한다. `re.compile` 함수를 사용해 패턴 객체를 만든다. 이때 함수 인자에는 내가 찾고자 하는 문자열 패턴에 대한 내용을 입력한다. `FILTERS` 변수는 물음표와 마침표를 포함해서 제거하고자 하는 기호의 집합을 정규 표현식으로 나타낸 문자열이다. 이렇게 정의한 패턴은 정규 표현식의 컴파일 함수를 사용해 컴파일해 둔다. 
- 그리고 데이터의 두 질문을 각 리스트로 만든 후, 각 리스트에 대해 전처리를 진행한다. 앞서 정의한 필터에 해당하는 문자열을 제거하고 모든 알파벳 문자를 소문자로 바꾼다. 이렇게 각 질문 리스트에 대해 전처리를 진행한 결과를 두 개의 변수에 저장한다.
- 이렇게 텍스트를 정제하는 작업을 끝냈다. 이제 남은 과정은 정제된 텍스트 데이터를 토크나이징하고 각 단어를 인덱스로 바꾼 후, 전체 데이터의 길이를 맞추기 위해 정의한 최대 길이보다 긴 문장은 자르고 짧은 문장은 패딩 처리를 하는 것이다.
- 문자열 토크나이징은 앞서와 동일한 방법으로 텐서플로 케라스에서 제공하는 자연어 전처리 모듈을 활용한다. 이때 4장과 다른 점은 토크나이징 객체를 만들 때는 두 질문 텍스트를 합친 리스트에 대해 적용하고, 토크나이징은 해당 객체를 활용해 각 질문에 대해 따로 진행한다는 것이다. 이러한 방법을 사용하는 이유는 두 질문에 대해 토크나이징 방식을 동일하게 진행하고, 두 질문을 합쳐 전체 단어 사전을 만들기 위해서다. 토크나이징 이후에는 패딩 처리를 한 벡터화를 진행할 것이다.

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_questions1 + filtered_questions2)
```

- 이렇게 생성한 토크나이징 객체를 두 질문 리스트에 적용해 각 질문을 토크나이징하고 단어들을 각 단어의 인덱스로 변환하자.

```python
questions1_sequence = tokenizer.texts_to_sequences(filtered_questions1)
questions2_sequence = tokenizer.texts_to_sequences(filtered_questions2)
```

- 단어 인덱스로 이뤄진 벡터로 바꾼 값을 확인해 보면 어떤 구조로 바뀌었는지 확인할 수 있을 것이다. 이제 모델에 적용하기 위해 특정 길이로 동일하게 맞춰야 한다. 따라서 최대 길이를 정한 후 그 길이보다 긴 질문은 자르고, 짧은 질문은 부족한 부분을 0으로 채우는 패딩 과정을 진행하자.

```python
MAX_SEQUENCE_LENGTH = 31

q1_data = pad_sequences(questions1_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
q2_data = pad_sequences(questions2_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
```

- 최대 길이의 경우 앞서 데이터 분석에서 확인했던 단어 개수 99퍼센트인 31로 설정했다. 이렇게 설정한 이유는 이상치를 뺀 나머지를 포함하기 위해서다(다양한 값으로 실험했을 때 이 값이 가장 좋은 값이었다). 전처리 모듈의 패딩 함수를 사용해 최대 길이로 자르고 짧은 데이터에 대해서는 데이터 뒤에 패딩값을 채워넣었다.
- 전처리가 끝난 데이터를 저장한다. 저장하기 전에 라벨값과 단어 사전을 저장하기 위해 값을 저장한 후 각 데이터의 크기를 확인해 보자.

```python
word_vocab = {}
word_vocab = tokenizer.word_index 
word_vocab["<PAD>"] = 0

labels = np.array(train_data['is_duplicate'], dtype=int)

print('Shape of question1 data: {}'.format(q1_data.shape))
print('Shape of question2 data:{}'.format(q2_data.shape))
print('Shape of label: {}'.format(labels.shape))
print("Words in index: {}".format(len(word_vocab)))
```

```
Shape of question1 data: (298526, 31)
Shape of question2 data:(298526, 31)
Shape of label: (298526,)
Words in index: 76504
```

- 두 개의 질문 문장의 경우 각각 길이를 31로 설정했고, 단어 사전의 길이인 전체 단어 개수는 76,605개로 돼 있다. 그리고 단어 사전과 전체 단어의 개수는 딕셔너리 형태로 저장해 두자.

```python
data_configs = {}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab)
```

- 이제 각 데이터를 모델링 과정에서 사용할 수 있게 저장하면 된다. 저장할 파일명을 지정한 후 각 데이터의 형태에 맞는 형식으로 저장하자.

```python
TRAIN_Q1_DATA = 'train_q1.npy'
TRAIN_Q2_DATA = 'train_q2.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
DATA_CONFIGS = 'data_configs.json'

np.save(open(DATA_IN_PATH + TRAIN_Q1_DATA, 'wb'), q1_data)
np.save(open(DATA_IN_PATH + TRAIN_Q2_DATA , 'wb'), q2_data)
np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA , 'wb'), labels)

json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'))
```

- 넘파이의 `save` 함수를 활용해 각 질문과 라벨 데이터를 저장한다. 딕셔너리 형태의 데이터 정보는 json 파일로 저장했다. 이렇게 하면 학습할 모델에 대한 데이터 전처리가 완료된다. 전처리한 데이터는 뒤에 모델 학습을 하는 과정에서 손쉽게 활용될 것이다. 이제 평가 데이터에 대해서도 앞의 전처리 과정을 동일하게 진행한 후 전처리한 데이터를 저장하자. 우선 전처리할 평가 데이터를 불러오자.

```python
test_data = pd.read_csv(DATA_IN_PATH + 'test.csv', encoding='utf-8')
# test_data = test_data.drop(test_data.tail(1217679).index,inplace=True) # drop last n rows
valid_ids = [type(x) ==int for x in test_data.test_id] 
test_data = test_data[valid_ids].drop_duplicates()
```

- 우선 평가 데이터에 대해 텍스트를 정제하자. 평가 데이터 역시 두 개의 질문이 존재한다. 따라서 각 질문을 따로 리스트로 만든 후 전처리할 것이다. 앞서 확인했듯이 평가 데이터의 길이가 학습 데이터와 비교했을 때 매우 길었다. 따라서 학습 데이터 때와 달리 시간이 많이 소요될 것이다.

```python
test_questions1 = [str(s) for s in test_data['question1']]
test_questions2 = [str(s) for s in test_data['question2']]

filtered_test_questions1 = list()
filtered_test_questions2 = list()

for q in test_questions1:
     filtered_test_questions1.append(re.sub(change_filter, "", q).lower())
        
for q in test_questions2:
     filtered_test_questions2.append(re.sub(change_filter, "", q).lower())
```

- 정제한 평가 데이터를 인덱스 벡터로 만든 후 동일하게 패딩 처리를 하면 된다. 이때 사용하는 토크나이징 객체는 이전에 학습 데이터에서 사용했던 객체를 사용해야 동일한 인덱스를 가진다. 

```python
test_questions1_sequence = tokenizer.texts_to_sequences(filtered_test_questions1)
test_questions2_sequence = tokenizer.texts_to_sequences(filtered_test_questions2)

test_q1_data = pad_sequences(test_questions1_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_q2_data = pad_sequences(test_questions2_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
```

- 평가 데이터의 경우 라벨이 존재하지 않으므로 라벨은 저장할 필요가 없다. 그리고 평가 데이터에 대한 단어 사전 정보도 이미 학습 데이터 전처리 과정에서 저장했기 때문에 추가로 저장할 필요가 없다. 하지만 평가 데이터에 대한 결과를 캐글에 제출할 때를 생각해보면 평가 데이터의 id 값이 필요하다. 따라서 평가 데이터의 id 값을 넘파이 배열로 만들자. 그리고 평가 데이터를 전처리한 값들의 크기를 출력해보자.

```python
test_id = np.array(test_data['test_id'])

print('Shape of question1 data: {}'.format(test_q1_data.shape))
print('Shape of question2 data:{}'.format(test_q2_data.shape))
print('Shape of ids: {}'.format(test_id.shape))
```

```
Shape of question1 data: (2345796, 31)
Shape of question2 data:(2345796, 31)
Shape of ids: (2345796,)
```

- 평가 데이터도 마찬가지로 전체 문장의 길이를 11로 맞춰서 전처리를 마무리했다. 이제 전처리한 평가 데이터를 파일로 저장하자. 두 개의 질문 데이터와 평가 id 값을 각각 넘파이 파일로 저장하자.

```
TEST_Q1_DATA = 'test_q1.npy'
TEST_Q2_DATA = 'test_q2.npy'
TEST_ID_DATA = 'test_id.npy'

np.save(open(DATA_IN_PATH + TEST_Q1_DATA, 'wb'), test_q1_data)
np.save(open(DATA_IN_PATH + TEST_Q2_DATA , 'wb'), test_q2_data)
np.save(open(DATA_IN_PATH + TEST_ID_DATA , 'wb'), test_id)
```

- 이제 모든 전처리 과정이 끝났다. 본격적으로 질문간의 유사도를 측정하기 위한 모델을 만들어 보자. 모델은 총 세 가지의 종류를 만들어 보겠다. 먼저 XG 부스트 모델을 만들어 텍스트 유사도 문제를 해결해 보자.

## 모델링

> 앞서 전처리한 데이터를 사용해 본격적으로 텍스트 유사도를 측정하기 위한 모델을 만들자. 여기서는 앞서 언급한 대로 총 세 개의 모델을 직접 구현하고 성능을 측정해서 모델 간의 성능을 비교해볼 것이다. 맨 처음 구현할 모델은 XG 부스트 모델이며, 나머지 두 개의 모델은 딥러닝 기반의 모델로서 하나는 합성곱 신경망을 활용한 모델이고 다른 하나는 맨해튼 거리를 활용하는 `LSTM` 모델인 `MaLSTM` 모델이다.

### XG 부스트 텍스트 유사도 분석 모델

- 맨 먼저 사용할 모델은 앙상블 모델 중 하나인 XG 부스트 모델이다. 해당 모델을 사용해서 데이터의 주어진 두 질문 문장 사이의 유사도를 측정해서 두 질문이 중복인지 아닌지를 판단할 수 있게 만들 것이다. 우선 XG 부스트 모델이 어떤 모델인지 먼저 알아보자.

#### 모델 소개

- XG 부스트란 `eXtream Gradient Boosting`의 약자로 최근 캐글 사용자에게 큰 인기를 얻고 있는 모델 중 하나다. XG 부스트는 앙상블의 한 방법인 부스팅(Boosting) 기법을 사용하는 방법이라서 XG 부스트에 대해 알아보기 전에 부스팅 기법에 대해 먼저 알아보자.
- 머신러닝 혹은 통계학에서 앙상블 기법이란 여러 개의 학습 알고리즘을 사용해 더 좋은 성능을 얻는 방법을 뜻한다. 앙상블 기법에는 배깅과 부스팅이라는 방법이 있다. 배깅에 대해 먼저 설명하면 배깅이란 여러 개의 학습 알고리즘, 모델을 통해 각각 결과를 예측하고 모든 결과를 동등하게 보고 취합해서 결과를 얻는 방식이다. 예를 들면 4장에서 사용했던 랜덤 포레스트의 경우 여러 개의 의사결정 트리 결괏값의 평균을 통해 결과를 얻는 배깅(Bagging)이라는 방법을 사용했다.
- 다음으로 부스팅에 대해 알아보자. 배깅의 경우 여러 알고리즘, 모델의 결과를 다 동일하게 취합한다고 했다. 이와 다르게 부스팅은 각 결과를 순차적으로 취합하는데, 단순히 하나씩 취합하는 방법이 아니라 이전 알고리즘, 모델이 학습 후 잘못 예측한 부분에 가중치를 줘서 다시 모델로 가서 학습하는 방식이다. 다음 그림을 보면 배깅과 부스팅에 대해 좀 더 직관적으로 이해할 수 있을 것이다.

![스크린샷 2025-04-08 오후 10 06 36](https://github.com/user-attachments/assets/43311549-8b4c-408e-9c5d-41a0e6cc3aaa)

- 그림에서 싱글이라고 나와 있는 부분은 앙상블 기법이 아니라 단순히 하나의 모델만으로 결과를 내는 방법이다. 앞에서 사용했던 `CNN`, `RNN` 등이 싱글에 해당한다. 
- 이렇게 해서 부스팅이 어떤 모델을 뜻하는지 알아봤다. XG 부스트는 부스팅 기법 중 트리 부스팅(Tree Boosting) 기법을 활용한 모델이다. 여기서 말하는 트리 부스팅이 무엇인지 알아보기 위해 4장에서 알아본 랜덤 포레스트를 생각해보자. 랜덤 포레스트 모델이란 여러 개의 의사결정 트리를 사용해 결과를 평균 내는 방법이라 배웠다. 따라서 랜덤 포레스트는 배깅에 해당하는 기법이었는데, 트리 부스팅은 동일한 원리에 부스팅 방식을 적용했다고 생각하면 된다. 즉, 트리 부스팅 기법이란 여러 개의 의사결정 트리를 사용하지만 단순히 결과를 평균내는 것이 아니라 결과를 보고 오답에 대해 가중치를 부여한다. 그리고 가중치가 적용된 오답에 대해서는 관심을 가지고 정답이 될 수 있도록 결과를 만들고 해당 결과에 대한 다른 오답을 찾아 다시 똑같은 작업을 반복적으로 진행하는 것이다.
- 최종적으로 XG 부스트란 이러한 트리 부스팅 방식에 경사 하강법을 통해 최적화하는 방법이다. 그리고 연산량을 줄이기 위해 의사결정 트리를 구성할 때 병렬 처리를 사용해 빠른 시간에 학습이 가능하다.
- 모델에 대한 설명을 들었을 때는 복잡해 보이고 사용하기 어려워 보일 수 있지만 이미 XG 부스트를 구현해둔 라이브러리를 사용하면 매우 쉽게 사용할 수 있다. 이제 모델을 사용하는 방법을 알아보자.

#### 모델 구현

- 이제 본격적으로 XG 부스트 모델을 직접 구현해 보자. 여기서는 사이킷런이나 텐서플로 라이브러리가 아닌 XG 부스트 모델만을 위한 라이브러리를 사용해서 구현하겠다. 우선 전처리한 데이터를 불러오자.

```python
import pandas as pd
import numpy as np
import os

import json


DATA_IN_PATH = './data_in/'

TRAIN_Q1_DATA_FILE = 'train_q1.npy'
TRAIN_Q2_DATA_FILE = 'train_q2.npy'
TRAIN_LABEL_DATA_FILE = 'train_label.npy'

# 훈련 데이터 가져오는 부분이다.
train_q1_data = np.load(open(DATA_IN_PATH + TRAIN_Q1_DATA_FILE, 'rb'))
train_q2_data = np.load(open(DATA_IN_PATH + TRAIN_Q2_DATA_FILE, 'rb'))
train_labels = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA_FILE, 'rb'))
```

- 넘파이 파일로 저장한 전처리 데이터를 불러왔다. 데이터의 경우 각 데이터에 대해 두 개의 질문이 주어져 있는 형태다. 현재는 두 질문이 따로 구성돼 있는데 이를 하나씩 묶어 하나의 질문 쌍으로 만들어야 한다. 

```python
train_input = np.stack((train_q1_data, train_q2_data), axis=1) 
```

- 넘파이의 stack 함수를 사용해 두 질문을 하나의 쌍으로 만들었다. 예를 들어, 질문 \[A\]와 질문 \[B\]가 있을 때 이 질문을 하나로 묶어 \[\[A\],\[B\]\] 형태로 만들었다. 이렇게 하나로 묶은 데이터 형태를 출력해 보자.

```python
print(train_input.shape)
```

```
(298526, 2, 31)
```

- 전체 29만 개 정도의 데이터에 대해 두 질문이 각각 31개의 질문 길이를 가지고 있음을 확인할 수 있다. 두 질문 쌍이 하나로 묶여 있는 것도 확인할 수 있다. 두 질문 쌍이 하나로 묶여 있는 것도 확인할 수 있다. 이제 학습 데이터의 일부를 모델 검증을 위한 검증 데이터로 만들어 두자.

```python
from sklearn.model_selection  import train_test_split

train_input, eval_input, train_label, eval_label = train_test_split(train_input, train_labels, test_size=0.2, random_state=4242)
```

- 전체 데이터의 20%를 검증 데이터로 만들어 뒀다. 이제 학습 데이터를 활용해 XG 부스트 모델을 학습시키고 검증 데이터를 활용해 모델의 성능을 측정해 보자. 모델을 구현하기 위해 `xgboost`라는 라이브러리를 활용할 것이다. 우선 해당 라이브러리가 설치돼 있지 않다면 설치하자. XG 부스트의 경우 공식 페이지의 설치 가이드를 참조해서 설치하면 된다. 
- XG 부스트를 설치한 후 라이브러리를 불러와서 모델을 구현해 보자. 모델에 적용하기 위해 입력값을 형식에 맞게 만들자.

```python
import xgboost as xgb

train_data = xgb.DMatrix(train_input.sum(axis=1), label=train_label) # 학습 데이터 읽어 오기
eval_data = xgb.DMatrix(eval_input.sum(axis=1), label=eval_label) # 평가 데이터 읽어 오기

data_list = [(train_data, 'train'), (eval_data, 'valid')]
```

- XG 부스트 모델을 사용하려면 입력값을 xgb 라이브러리의 데이터 형식인 `DMatrix` 형태로 만들어야 한다. 학습 데이터와 검증 데이터 모두 적용해서 해당 데이터 형식으로 만든다. 적용 과정에서 각 데이터에 대해 `sum` 함수를 사용하는데 이는 각 데이터의 두 질문을 하나의 값으로 만들어 주기 위해서다. 그리고 두 개의 데이터를 묶어서 하나의 리스트로 만든다. 이때 학습 데이터와 검증 데이터는 각 상태의 문자열과 함께 듀플 형태로 구성한다.
- 이제 모델을 생성하고 학습하는 과정을 진행해 보자.

```python
params = {} # 인자를 통해 XGB모델에 넣어 주자 
params['objective'] = 'binary:logistic' # 로지스틱 예측을 통해서 
params['eval_metric'] = 'rmse' # root mean square error를 사용  

bst = xgb.train(params, train_data, num_boost_round = 1000, evals = data_list, early_stopping_rounds=10)
```

- 우선 모델을 만들고 학습하기 위해 몇 가지 선택해야 하는 옵션은 딕셔너리를 만들어 넣으면 된다. 이때 딕셔너리에는 모델의 목적함수와 평가지표를 정해서 넣어야 하는데 여기서는 우선 목적함수의 경우 이진 로지스틱 함수를 사용한다. 평가 지표의 경우 rmse(Root mean squared error)를 사용한다. 이렇게 만든 인자와 학습 데이터, 데이터를 반복하는 횟수인 `num_boost_round`, 모델 검증 시 사용할 전체 데이터 쌍, 그리고 조기 멈춤(`early stopping`)을 위한 횟수를 지정한다. 
- 데이터를 반복하는 횟수, 즉 에폭을 의미하는 값으로 1000을 설정했다. 전체 데이터를 첫번 반복해야 끝나도록 설정한 것이다. 그리고 조기 멈춤을 위한 횟수값으로 10을 설정했는데 이는 만약 10에폭 동안 에러값이 별로 줄어들지 않았을 경우에는 학습을 조기에 멉추게 하는 것이다. 이렇게 설정하고 함수를 실행하면 다음과 같이 학습이 진행되고 여러 값을 확인할 수 있을 것이다. 


![스크린샷 2025-04-09 오후 9 47 06](https://github.com/user-attachments/assets/ce9f8d2b-efd4-40a8-8b83-6a910475ee19)

- 각 스텝마다 학습 에러와 검증 에러를 계속해서 보여주고 있으며, 877 스텝에서 학습이 끝났다. 이는 더 이상 에러가 떨어지지 않아서 학습이 조기 멈춤한 것이다. 이렇게 학습한 모델을 사용해 평가 데이터를 예측하고 예측 결과를 캐글에 제출할 수 있게 파일로 만들어보자. 우선 전처리한 평가 데이터를 불러오자.

```python
TEST_Q1_DATA_FILE = 'test_q1.npy'
TEST_Q2_DATA_FILE = 'test_q2.npy'
TEST_ID_DATA_FILE = 'test_id.npy'

test_q1_data = np.load(open(DATA_IN_PATH + TEST_Q1_DATA_FILE, 'rb'))
test_q2_data = np.load(open(DATA_IN_PATH + TEST_Q2_DATA_FILE, 'rb'))
test_id_data = np.load(open(DATA_IN_PATH + TEST_ID_DATA_FILE, 'rb'))
```

- 불러온 평가 데이터를 앞의 학습 데이터와 마찬가지로 XG 부스트 모델에 적용할 수 있게 형식에 맞춰 만든 후 모델의 `predict` 함수에 적용한다.

```python
test_input = np.stack((test_q1_data, test_q2_data), axis=1) 
test_data = xgb.DMatrix(test_input.sum(axis=1))
test_predict = bst.predict(test_data)
```

- 이렇게 예측한 결괏값을 형식에 맞게 파일로 만들어야 한다. 평가 데이터의 id 값과 예측값을 하나의 데이터프레임으로 만든 후 CSV 파일로 저장하자. 

```python
DATA_OUT_PATH = './data_out/'

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
    
output = pd.DataFrame({'test_id': test_id_data, 'is_duplicate': test_predict})
output.to_csv(DATA_OUT_PATH + 'simple_xgb.csv', index=False)
```

- 지정한 경로에 CSV 파일이 만들어졌을 것이다. 이제 해당 파일을 캐글에 제출해서 점수를 확인해 보자.
- XG 부스트를 통해 점수를 받아봤다. 순위를 확인해 보면 중간 정도의 순위다. 간단하게 구현했지만 성능을 더 올리고 싶다면 `TF-IDF`나 `word2vec`으로 데이터의 입력값의 형태를 바꾼 후 모델에 적용하는 방법을 추천한다. 이제 딥러닝 모델을 통해 유사도 문제를 해결해 보자.

### CNN 텍스트 유사도 분석 모델

- 앞에서는 머신러닝 모델 중 하나인 XG 부스트를 이용해 텍스트 유사도를 측정했다. 이번에는 딥러닝 모델을 만들어 보겠다. 그중에서 4장에서도 사용했던 합성곱 신경망 구조를 활용해 텍스트 유사도를 측정하는 모델을 만들어 보겠다. 기본적인 구조는 이전 장의 합성곱 모델과 유사하지만 이번 경우에는 각 데이터가 두 개의 텍스트 문장으로 돼 있기 때문에 병렬적인 구조를 가진 모델을 만들어야 한다. 본격적으로 모델이 어떻게 구성되고 어떻게 구현하는지 알아보자.

#### 모델 소개

- CNN 텍스트 유사도 분석 모델은 문장에 대한 의미 벡터를 합성곱 신경망을 통해 추출해서 그 벡터에 대한 유사도를 측정한다.

