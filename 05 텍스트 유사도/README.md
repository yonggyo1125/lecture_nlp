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



