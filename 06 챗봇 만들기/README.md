# 챗봇 만들기

- 지금까지 두 가지 문제에 대해 실습을 진행했다. 4장에서는 텍스트를 분석해서 각 텍스트를 분류하는 문제를 실습했고, 5장에서는 두 개의 텍스트가 있을 때 각 텍스트끼리의 유사도를 판단하는 문제를 실습했다. 이번 장에서는 텍스트를 단순히 분석해서 분류나 유사도를 측정하는 것이 아닌 직접 문장을 생성할 수 있는 텍스트 생성(text generation) 문제를 실습해 보겠다. 텍스트 생성에도 많은 문제가 있지만 '자연어의 꽃'이라고 불리는 '챗봇'을 제작해 본다.
- 일반적으로 챗봇을 제작하는 방법은 매우 다양하다. 단순하게 규칙 기반으로 제작할 수도 있고, 머신러닝을 활용한 유사도 기반, 규칙과 머신러닝을 섞은 하이브리드 기반, 특정 시나리오 기반까지 정의하는 사람에 따라 제작 방법이 매우 다양하다. 이 책에서는 이러한 제작 방법 중에서 딥러닝 모델을 통한 챗봇을 만들어 보겠다. 또한 챗봇을 만들기 위한 딥러닝 모델에도 여러 가지가 있지만 그중에서 번역 문제에서 이미 성능이 입증된 시퀀스 투 시퀀스(Sequence to sequence) 모델을 활용해 챗봇을 제작하는 법을 알아보고자 한다.
- 하지만 모든 딥러닝 모델이 그렇듯 데이터가 있어야 모델을 학습할 수 있다. 따라서 한글로 챗봇을 만들기 위한 데이터에 대해 먼저 알아본다.

## 데이터 소개

|             |                                        |
| ----------- | -------------------------------------- |
| 데이터이름  | Chatbot data                           |
| 데이터 용도 | 한국어 챗봇 학습을 목적으로 사용한다.  |
| 데이터 권한 | MIT 라이센스                           |
| 데이터 출처 | https://github.com/songys/Chatbot_data |

- 일반적으로 공개된 챗봇을 위한 한글 데이터는 거의 없다고 봐도 무방하다. 심지어 한글보다 많은 데이터가 공개돼 있는 영어에도 'Ubuntu Dialogue Corpus' 데이터를 제외하면 공개된 데이터가 없다고 볼 수 있다. 다행히 한글로도 챗봇을 만들어 볼 수 있게 데이터를 제작해서 공개해주신 분들이 있다. 여기서 사용할 데이터는 송영숙님이 번역 및 제공해 주신 'Chatbot_data_for_Korean v1.0' 데이터셋이다.
- 이 데이터는 총 11,876개의 데이터로 구성돼 있고, 각 데이터는 질문과 그에 대한 대답, 그리고 주제에 대한 라벨값을 가지고 있다. 이 라벨값은 3가지로 구성돼 있는데 0은 일상 대화를 나타내고, 1은 부정, 2는 긍정의 주제를 의미한다. 앞서 다뤘던 데이터에 비하면 적은 수의 데이터이지만 10,000개가 넘는 데이터이기 때문에 연구 및 공부하기에는 충분한 양이다. 따라서 이 데이터를 사용해 텍스트 생성 문제, 그중에서도 챗봇을 직접 만들어 보자.

## 데이터 분석

- 이번 절에서는 실습을 진행하는 구성이 다른 장과 조금 다르다. 이전 장까지는 데이터에 대한 분석과 전처리를 진행한 후 전처리한 데이터를 가지고 여러 가지 모델링을 해봤다면 이번 장에서는 데이터 분석을 우선적으로 진행한 후 데이터 전처리와 모델을 한 번에 만들 것이다. 데이터 분석을 통해 나온 결과를 활용해 전처리 모듈을 만들어 보겠다.
- 우선 챗봇 데이터를 분석해서 데이터의 고유한 특징을 파악한 후 모델링 과정에서 고려해야 할 사항을 확인해보자. 전체적인 데이터 분석 과정은 이전에 진행했던 것과 유사하게 진행한다. 추가로 특정 질문을 했을 때 어떤 응답이 나올 수 있는지도 유추해 보자.
- 먼저 데이터 분석을 위해 데이터를 불러온다. 판다스 라이브러리를 사용해 데이터프레임 형태로 데이터를 불러오자.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from konlpy.tag import Okt

from functools import reduce
from wordcloud import WordCloud

DATA_IN_PATH = './data_in/'

data = pd.read_csv(DATA_IN_PATH + 'ChatBotData.csv', encoding='utf-8')
```

- 데이터를 불러오는 방법은 이전 방법과 동일하다. 데이터의 구조를 확인해 보기 위해 head 함수를 사용해 데이터의 일부만 출력해 보자.

```python
print(data.head())
```

|     | Q                         | A                   | label |
| --- | ------------------------- | ------------------- | ----- |
| 0   | 12시 땡!                  | 하루가 또 가네요    | 0     |
| 1   | 1지망 학교 떨어졌어       | 위로해 드립니다.    | 0     |
| 2   | 3박4일 놀러가고 싶다      | 여행은 언제나 좋죠. | 0     |
| 3   | 3박4일 정도 놀러가고 싶다 | 여행은 언제나 좋죠. | 0     |
| 4   | PPL 심하네                | 눈살이 찌푸려지죠.  | 0     |

- 데이터는 앞서 말했던 것과 동일한 구조로 돼 있다. 각 데이터는 Q, A 값인 질문과 대답 텍스트를 가지고 있고 그에 대한 라벨값을 가지고 있다. 해당 데이터에서 라벨값은 0,1,2로 구성돼 있다. 이제 데이터를 좀 더 깊이 있게 분석해 보자.

### 문장 전체에 대한 분석

- 먼저 데이터의 길이를 분석한다. 질문과 답변 모두에 대해 길이를 분석하기 위해 두 데이터를 하나의 리스트로 만들자.

```python
sentences = list(data['Q']) + list(data['A'])
```

- 질문과 답변 문장을 위와 같이 하나의 리스트로 만들었다면 이제 길이를 분석한다. 이전 장까지 2개의 기준으로 분석을 진행했다. 문자 단위의 길이 분석과 단어의 길이 분석을 진행했는데, 이번 장에서는 하나의 기준을 추가해서 세 가지 기준으로 분석을 진행한다. 분석 기준은 다음과 같다.

  - 분자 단위의 길이 분석(음절)
  - 단어 단위의 길이 분석(어절)
  - 형태소 단위의 길이 분석

- 음절의 경우 문자 하나하나를 생각하면 된다. 어절의 경우 간단하게 띄어쓰기 단위로 생각하면 된다. 마지막으로 형태소 단위의 경우, 어절과 음절 사이라고 생각하면 된다. 여기서 형태소란 의미를 가지는 최소 단위를 의미한다. 예를 들어 이해해 보자. 다음과 같은 문장이 있다고 하자.

```
"자연어 처리 공부는 매우 어렵다"
```

- 이 문장을 각각 음절, 어절, 형태소 단위로 나눈 결과와 각 길이는 다음과 같다.
  - 음절: "자","연","어","처","리","공","부","는","매","우","어","렵","다"(길이: 13)
  - 어절: "자연어","처리","공부는","매우","어렵다"(길이: 5)
  - 형태소: "자연어","처리","공부","는","매우","어렵","다"(길이: 7)
- 이처럼 형태소로 나눴을 때가 단순히 띄어쓰기로 구분해서 나눴을 때보다 좀 더 의미를 잘 표현한다고 볼 수 있다. 이처럼 세 가지 기준으로 나눈 후 각 길이를 측정하겠다.
- 먼저 각 기준에 따라 토크나이징해 보자. 형태소의 경우 토크나이징을 위해 2장에서 알아본 `KoNLPy'를 사용한다.

```python
tokenized_sentences = [s.split() for s in sentences]
sent_len_by_token = [len(t) for t in tokenized_sentences]
sent_len_by_eumjeol = [len(s.replace(' ', '')) for s in sentences]

okt = Okt()

morph_tokenized_sentences = [okt.morphs(s.replace(' ', '')) for s in sentences]
sent_len_by_morph = [len(t) for t in morph_tokenized_sentences]
```

- 우선 띄어쓰기 기준으로 문장을 나눈다. 이 값의 길이를 측정해서 어절의 길이를 측정하고 이 값을 다시 붙여서 길이를 측정해서 음절의 길이로 사용한다. 마지막으로 형태소로 나누기 위해 `KoNLPy`에 `Okt` 형태소 분석기를 사용해서 나눈 후 길이를 측정한다.
- 이렇게 각 기준으로 나눈 후 길이를 측정한 값을 각각 변수로 설정해두면 이 값을 사용해 그래프를 그리거나 각종 통곗값을 측정할 수 있다. 맷플롯립을 활용해 각각에 대한 그래프를 그려보자.

```python
plt.figure(figsize=(12, 5))
plt.hist(sent_len_by_token, bins=50, range=[0,50], alpha=0.5, color= 'r', label='eojeol')
plt.hist(sent_len_by_morph, bins=50, range=[0,50], alpha=0.5, color='g', label='morph')
plt.hist(sent_len_by_eumjeol, bins=50, range=[0,50], alpha=0.5, color='b', label='eumjeol')
plt.title('Sentence Length Histogram')
plt.xlabel('Sentence Length')
plt.ylabel('Number of Sentences')
```

- 그래프의 경우 세 가지 기준으로 구한 각 길이를 한번에 그려본다. 맷플롯립을 사용해 각 히스토그램을 정의한다. 이때 구분을 위해서 각 색을 구분지어서 설정하자. 그리고 이 그래프의 제목과 x축, y축 이름을 설정한 후 그래프를 그려보면 다음과 같이 나올 것이다.

![스크린샷 2025-03-28 오후 9 29 07](https://github.com/user-attachments/assets/bff85033-e5f3-4598-ad7c-a4846be2daf0)

- 그래프를 보면 빨간색 히스토그램은 어절 단위에 대한 히스토그램이고, 형태소의 경우는 초록색이고, 음절의 경우는 파란색이다. 그래프 결과를 보면 어절이 가장 길이가 낮은 분포를 보이고, 그다음으로는 형태소, 가장 긴 길이를 가지고 있는 것이 음절 단위다. 하지만 이러한 분석은 어떤 텍스트 데이터를 사용하더라도 당연한 결과다. 그리고 히스토그램을 통해 각 길이가 어느 쪽으로 치우쳐 있는지 혹은 각 데이터에 이상치는 없는지 등도 확인할 수 있는데 이 히스토그램을 통해서는 직관적으로 확인하기 어렵다. 이는 각 히스토그램의 y값의 분포가 매우 다르기 때문인데, y값의 크기를 조정함으로써 이를 해결할 수 있다. 위의 히스토그램 코드를 다음과 같이 수정한 후 결과를 확인해 보자.

```python
plt.figure(figsize=(12, 5))
plt.hist(sent_len_by_token, bins=50, range=[0,50], alpha=0.5, color= 'r', label='eojeol')
plt.hist(sent_len_by_morph, bins=50, range=[0,50], alpha=0.5, color='g', label='morph')
plt.hist(sent_len_by_eumjeol, bins=50, range=[0,50], alpha=0.5, color='b', label='eumjeol')
plt.yscale('log')
plt.title('Sentence Length Histogram by Eojeol Token')
plt.xlabel('Sentence Length')
plt.ylabel('Number of Sentences')
```

- 이전 코드와 달라진 점은 중간에 yscale 함수를 사용했다는 점이다. 함수의 인자로 사용된 `log`는 각 그래프가 가지는 y값의 스케일을 조정함으로써 차이가 큰 데이터에 대해서도 함께 비교할 수 있게 한다. 그래프는 다음과 같이 그려질 것이다.

![스크린샷 2025-03-28 오후 9 33 39](https://github.com/user-attachments/assets/1a9e94e9-50ab-4ef8-bf2b-dca93115fca0)

- 히스토그램의 y 값의 스케일을 조정한 그래프를 보면 이전에는 보이지 않았던 분포의 꼬리 부분이 어떻게 분포돼 있었는지 보기 쉽게 나온다. 어절의 경우 길이가 20인 경우가 이상치 데이터로 존재하고 형태소나 음절의 경우 각각 30, 45 정도 길이에서 이상치가 존재한다. 이러한 길이 분포에 대한 분석 내용을 바탕으로 입력 문장의 길이를 어떻게 설정할지 정의하면 된다.
- 이제 각 길이값을 히스토그램이 아닌 정확한 수치를 확인하기 위해 각 기준별 길이에 대한 여러 가지 통곗값을 확인해 보자. 우선 어절에 대해 각 통계값을 출력해 보자.

```python
print('어절 최대길이: {}'.format(np.max(sent_len_by_token)))
print('어절 최소길이: {}'.format(np.min(sent_len_by_token)))
print('어절 평균길이: {:.2f}'.format(np.mean(sent_len_by_token)))
print('어절 길이 표준편차: {:.2f}'.format(np.std(sent_len_by_token)))
print('어절 중간길이: {}'.format(np.median(sent_len_by_token)))
print('제 1 사분위 길이: {}'.format(np.percentile(sent_len_by_token, 25)))
print('제 3 사분위 길이: {}'.format(np.percentile(sent_len_by_token, 75)))
```

```
어절 최대 길이: 21
어절 최소 길이: 1
어절 평균 길이: 3.64
어절 길이 표준편차: 1.74
어절 중간 길이: 3.0
제1사분위 길이: 2.0
제3사분위 길이: 5.0
```

- 어절로 나눈 길이의 통곗값은 위와 같이 확인할 수 있다. 해당 통곗값은 앞서 진행한 데이터 분석과 동일하다. 이제 이 통곗값들을 어절뿐만 아니라 음절, 형태소 단위로 나눈 길이값에 대해서도 확인해 보자.

```python
print('형태소 최대길이: {}'.format(np.max(sent_len_by_morph)))
print('형태소 최소길이: {}'.format(np.min(sent_len_by_morph)))
print('형태소 평균길이: {:.2f}'.format(np.mean(sent_len_by_morph)))
print('형태소 길이 표준편차: {:.2f}'.format(np.std(sent_len_by_morph)))
print('형태소 중간길이: {}'.format(np.median(sent_len_by_morph)))
print('형태소 1/4 퍼센타일 길이: {}'.format(np.percentile(sent_len_by_morph, 25)))
print('형태소 3/4 퍼센타일 길이: {}'.format(np.percentile(sent_len_by_morph, 75)))
```

```python
print('음절 최대길이: {}'.format(np.max(sent_len_by_eumjeol)))
print('음절 최소길이: {}'.format(np.min(sent_len_by_eumjeol)))
print('음절 평균길이: {:.2f}'.format(np.mean(sent_len_by_eumjeol)))
print('음절 길이 표준편차: {:.2f}'.format(np.std(sent_len_by_eumjeol)))
print('음절 중간길이: {}'.format(np.median(sent_len_by_eumjeol)))
print('음절 1/4 퍼센타일 길이: {}'.format(np.percentile(sent_len_by_eumjeol, 25)))
print('음절 3/4 퍼센타일 길이: {}'.format(np.percentile(sent_len_by_eumjeol, 75)))
```

|        | 최대 | 최소 | 평균  | 표준편차 | 중간값 | 제1사분위 | 제3사분위 |
| ------ | ---- | ---- | ----- | -------- | ------ | --------- | --------- |
| 어절   | 21   | 1    | 3.64  | 1.74     | 3      | 2         | 5         |
| 형태소 | 33   | 1    | 6.88  | 3.08     | 6      | 5         | 8         |
| 음절   | 57   | 1    | 11.31 | 4.98     | 10     | 8         | 14        |

- 각 길이에 대해 확인한 결과를 정리하면 위와 같은 결과가 나올 것이다. 평균값을 확인해 보면 우선 전체 문자 수는 11개 정도의 평균값을 가지고 있고, 띄어쓰기로 구분한 어절의 경우 각 문장당 3\~4 정도의 평균값을 보인다. 형태소로 분석한 경우 이보다 조금 더 큰 6\~7 정도의 평균값을 가지고 있다.
- 이제는 전체 데이터를 한번에 보기 쉽게 박스 플롯으로 그려보자. 박스 플롯에서도 3개의 기준을 모두 한 번에 확인한다.

```python
plt.figure(figsize=(12, 5))
plt.boxplot([sent_len_by_token, sent_len_by_morph, sent_len_by_eumjeol],
            labels=['Eojeol', 'Morph', 'Eumjeol'],
            showmeans=True)
```

- 여러 가지 값에 대해 한 번에 박스 플롯을 그리면 인자로 각 값들을 리스트를 만들어 넣으면 된다. 각 박스 플롯의 제목 역시 마찬가지로 리스트로 제목을 넣으면 된다. 이제 그려진 박스 플롯을 확인해 보자.

![스크린샷 2025-03-28 오후 10 02 20](https://github.com/user-attachments/assets/922ba92d-ecf5-4595-8cbc-f6f702dafd0b)

- 데이터 분포를 나타낸 박스 플롯을 보면 꼬리가 긴 형태로 분포돼 있음을 확인할 수 있다. 대체로 문장의 길이는 5\~15의 길이를 중심으로 분포를 이루고 있고 음절의 경우 길이 분포가 어절과 형태소에 비해 훨씬 더 크다는 점을 알 수 있다.
- 지금은 질문과 답변을 모두 합쳐서 데이터 전체 문장의 길이 분포를 확인했다. 그런데 앞으로 만들 모델의 경우 질문이 입력으로 들어가는 부분과 답변이 입력으로 들어가는 부분이 따로 구성돼 있다. 따라서 이번에는 질문과 답변을 구분해서 분석해 보자.

### 질문, 답변 각각에 대한 문장 길이 분포 분석

- 이제 전체 데이터가 아닌 질문과 응답으로 구성된 각 문장에 대한 길이 분포를 따로 알아보자. 앞서 길이를 분석할 때는 음절, 어절, 형태소 단위로 구분해서 분석했다. 여기서는 그 중에서 형태소 기준으로만 길이를 분석해 본다.

```python
query_sentences = list(data['Q'])
answer_sentences = list(data['A'])

query_morph_tokenized_sentences = [okt.morphs(s.replace(' ', '')) for s in query_sentences]
query_sent_len_by_morph = [len(t) for t in query_morph_tokenized_sentences]

answer_morph_tokenized_sentences = [okt.morphs(s.replace(' ', '')) for s in answer_sentences]
answer_sent_len_by_morph = [len(t) for t in answer_morph_tokenized_sentences]
```

- 우선은 데이터프레임의 질문 열과 답변 열을 각각 리스트로 정의한 후 앞서 진행한 것과 동일하게 `KoNLPy`의 `Okt` 형태소 분석기를 사용해 토크나이징한 후 구분된 데이터의 길이를 하나의 변수로 만든다. 이 과정을 질문과 답변에 대해 모두 진행했다. 이제 이렇게 형태소로 나눈 길이를 히스토그램으로 그려보자. 질문과 답변에 대한 길이를 한 번에 히스토그램으로 그린다.

```python
plt.figure(figsize=(12, 5))
plt.hist(query_sent_len_by_morph, bins=50, range=[0,50], color='g', label='Query')
plt.hist(answer_sent_len_by_morph, bins=50, range=[0,50], color='r', alpha=0.5, label='Answer')
plt.legend()
plt.title('Query Length Histogram by Morph Token')
plt.xlabel('Query Length')
plt.ylabel('Number of Queries')
```

![스크린샷 2025-03-28 오후 10 29 31](https://github.com/user-attachments/assets/07b48a27-453c-4b11-a52e-a7d338fe8b9c)

- 히스토그램을 살펴보면 전체적으로 질문 문장 길이가 응답 문장 길이보다 상대적으로 짧다는 것을 확인할 수 있다. 앞서 했던 것과 동일하게 해당 길이에 대해서도 이상치를 잘 확인할 수 있게 y값의 크기를 조정해서 다시 히스토그램을 그려보자.

```python
plt.figure(figsize=(12, 5))
plt.hist(query_sent_len_by_morph, bins=50, range=[0,50], color='g', label='Query')
plt.hist(answer_sent_len_by_morph, bins=50, range=[0,50], color='r', alpha=0.5, label='Answer')
plt.legend()
plt.yscale('log', nonposy='clip')
plt.title('Query Length Log Histogram by Morph Token')
plt.xlabel('Query Length')
plt.ylabel('Number of Queries')
```

![스크린샷 2025-03-28 오후 10 32 01](https://github.com/user-attachments/assets/e8470195-183d-455b-8813-af2a9de416d4)

- 답변 데이터가 질문 데이터보다 좀 더 이상치 값이 많은 것을 확인할 수 있다. 상대적으로 질문의 경우 평균 주변에 잘 분포돼 있음을 확인할 수 있다. 이 두 데이터에 대해 정확한 평균값을 확인하려면 다음과 같은 결과가 나올 것이다.

|             | 최대 | 최소 | 평균 | 표준편차 | 중간값 | 제1사분위 | 제3사분위 |
| ----------- | ---- | ---- | ---- | -------- | ------ | --------- | --------- |
| 질문 데이터 | 23   | 1    | 6.09 | 2.88     | 6.0    | 4         | 8         |
| 답변 데이터 | 33   | 1    | 7.67 | 3.08     | 7      | 6         | 9         |

- 통곗값을 확인해 보면 최댓값의 경우 답변 데이터가 훨씬 크다는 것을 확인할 수 있다. 그리고 평균의 경우에도 앞서 확인한 것과 같이 질문 데이터가 좀 더 작은 값을 보인다. 이제 두 데이터를 박스 플롯으로 그려보자

```python
query_sentences = list(data['Q'])
answer_sentences = list(data['A'])

query_morph_tokenized_sentences = [okt.morphs(s.replace(' ', '')) for s in query_sentences]
query_sent_len_by_morph = [len(t) for t in query_morph_tokenized_sentences]

answer_morph_tokenized_sentences = [okt.morphs(s.replace(' ', '')) for s in answer_sentences]
answer_sent_len_by_morph = [len(t) for t in answer_morph_tokenized_sentences]
```

```python
plt.figure(figsize=(12, 5))
plt.boxplot([query_sent_len_by_morph, answer_sent_len_by_morph], labels=['Query', 'Answer'])
```

![스크린샷 2025-03-28 오후 10 42 29](https://github.com/user-attachments/assets/616aa45e-d720-4bfa-af2f-c271065807ce)

- 박스 플롯을 보면 질문 데이터와 응답 데이터에 대한 분포가 앞서 그렸던 히스토그램과 통곗값에서 본 것과 조금은 다른 모습을 확인할 수 있다. 통계값에서는 답변 데이터에 대한 평균 길이가 질문 데이터보다 길었는데, 박스 플롯에 그려진 박스의 경우 질문 데이터가 더 큰 것을 확인할 수 있다. 즉, 답변 데이터의 경우 길이가 긴 이상치 데이터가 많아 평균값이 더욱 크게 측정됐다는 것을 확인할 수 있다. 이는 두 데이터가 전체적으로 차이가 난다고 하기보다는 답변 데이터에 이상치 데이터가 많아서 이렇게 측정된 것으로 해석할 수 있다.
- 이제 이 길이값을 통해 모델에 적용될 문장의 최대 길이를 결정해야 한다. 위에 나온 문장 길이에 대한 통계를 보고 중간값이나 제3사분위에 값을 적용할 수도 있다. 하지만 실제로 통계를 반영한 길이를 그대로 넣었을 때 만족할 만한 성능을 얻기는 쉽지 않았다. 디코더의 경우 문장의 뒷부분이 일부 잘려서 생성하고자 하는 문장이 완전한 문장이 아닌 문제가 있다는 것을 확인했다. 모델 학습 시도를 여러 번 한 끝에 이 책에서는 경험적으로 좋은 성능이 나올 수 있는 문장 길이를 25로 설정했다. 이 길이는 문장 길이 3사분위값 주변을 탐색하면서 가장 문장 생성를 잘 할 수 있는 길이를 찾아본 결과다.

### 데이터 어휘 빈도 분석

- 이때까지는 데이터의 길이 부분에 대해 분석을 진행했다. 이제는 데이터에서 사용되는 단어에 대해 분석해 보자. 어떤 단어가 사용되는지, 자주 사용되는 단어는 어떤 것들이 있는지 알아보겠다.
- 이제 형태소 단위로 토크나이징한 데이터를 사용해서 자주 사용하는 단어를 알아보자. 단순히 토크나이징한 데이터에서 자주 사용되는 단어를 분석하면 결과는 '이', '가', '를' 등의 조사가 가장 큰 빈도수를 차지할 것이다. 이는 어떤 데이터이든 당연한 결과이므로 분석하는 데 크게 의미가 없다. 따라서 의미상 중요한 명사, 형용사, 동사만 따로 모은 후 빈도수 분석을 진행한다.
- 먼저 품사에 따라 구분해야 명사, 형용사, 동사를 따로 모을 수 있는데, 품사를 확인하는 방법은 `KoNLPy`의 품사 분류(POS-tagging) 모듈을 사용하면 된다. 앞서 사용한 `Okt` 형태소 분석기의 품사 분류 기능을 사용하면 다음과 같이 결과가 나온다.

```python
okt.pos('오늘밤은유난히덥구나')
```

```
[('오늘밤', 'Noun'), ('은', 'Josa'), ('유난히', 'Adverb'), ('덥구나', 'Adjective')]
```

- 예시 문장인 '오늘밤은유난히덥구나'를 `Okt` 형태소 분석기를 사용해서 품사를 확인해 보면 위와 같이 결과가 나온다. 보다시피 각 형태소와 그에 해당하는 품사가 나오는데, 여기서는 명사, 형용사, 동사만 사용한다고 했으므로 `Noun`, `Adjective`, `Verb`만 사용하면 된다. 이제 각 문장에서 명사, 형용사, 동사를 제외한 단어를 모두 제거한 문자열을 만들어보자.

```python
uery_NVA_token_sentences = list()
answer_NVA_token_sentences = list()

for s in query_sentences:
    for token, tag in okt.pos(s.replace(' ', '')):
        if tag == 'Noun' or tag == 'Verb' or tag == 'Adjective':
            query_NVA_token_sentences.append(token)

for s in answer_sentences:
    temp_token_bucket = list()
    for token, tag in okt.pos(s.replace(' ', '')):
        if tag == 'Noun' or tag == 'Verb' or tag == 'Adjective':
            answer_NVA_token_sentences.append(token)

query_NVA_token_sentences = ' '.join(query_NVA_token_sentences)
answer_NVA_token_sentences = ' '.join(answer_NVA_token_sentences)
```

- 이처럼 간단히 전처리하고 나면 동사, 명사, 형용사를 제외한 나머지 문자는 모두 제거된 상태의 문자열이 만들어질 것이다. 이 문자열을 사용해서 어휘 빈도 분석을 진행한다. 앞서 4, 5장에서 사용했던 워드클라우드를 사용해 데이터의 어휘 빈도를 분석할 것이다. 한글 데이터를 워드클라우드로 그리기 위해서는 추가로 한글 폰트를 설정해야 한다.

```python
query_wordcloud = WordCloud(font_path= DATA_IN_PATH + 'NanumGothic.ttf').generate(query_NVA_token_sentences)

plt.imshow(query_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

![스크린샷 2025-03-28 오후 10 56 13](https://github.com/user-attachments/assets/c1b5abfe-603f-416f-a308-accdedf30d7c)

- 답변 데이터에 대해서도 동일하게 워드클라우드를 그려서 살펴보자.

```python
query_wordcloud = WordCloud(font_path= DATA_IN_PATH + 'NanumGothic.ttf').generate(answer_NVA_token_sentences)

plt.imshow(query_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

![스크린샷 2025-03-28 오후 10 56 25](https://github.com/user-attachments/assets/dc094c87-1118-4ae5-9c35-39a617c47481)

- 결과를 보면 응답 데이터에 대한 워드클라우드도 마찬가지로 비슷하게 연애에 관한 단어가 나온다. 질문 데이터와 다른 점은 '시간', '마음', '생각'과 같은 단어들이 더 많이 나온다는 것이다. 그리고 답변의 경우 대부분 권유의 문자열을 담고 있음을 유추할 수 있다.
- 이제 모든 데이터 분석 과정이 끝났다. 분석한 결과를 토대로 데이터를 전처리하고 모델을 만들 차례다. 좀 더 상세한 데이터 분석이 필요하다면 데이터를 직접 하나씩 무작위로 확인하는 것도 도움이 될 수 있다.

## 시퀀스 투 시퀀스 모델

- 이제 본격적으로 모델을 만들어 보겠다. 코드의 구성이 이전 장에서 진행했던 실습과 조금 다른데, 전처리 모듈화해서 파이썬 파일로 만들어 사용할 것이다. 따라서 훨씬 어렵게 느겨질 수 있는데, 객체지향적으로 코드를 구성한다면 이후 수정이나 모델의 재사용이 용이하므로 이와 같이 만드는 것이 향후 모델을 만드는 데 크게 도움이 될 것이다. 그렇가면 이번 장에서 사용할 모델에 대해 알아보자.

### 모델 소개
- 챗봇(대화 모델)을 만들기 위해 사용할 모델은 시퀀스 투 시퀀스(Sequence to Sequence)모델이다. 이 모델은 이름 그대로 시퀀스 형태의 입력값을 시퀀스 형태로 출력을 만들 수 있게 하는 모델이다. 즉, 하나의 텍스트 문장이 입력으로 들어오면 하나의 텍스트 문장을 출력하는 구조다. 이 모델이 가장 많이 활용되는 분야는 기계 번역(Machine translation) 분야이지만 텍스트 요약(Text summarization), 이미지 설명(Image captioning), 대화 모델(Conversation model) 등 다양한 분야에서 활용되고 있다.
- 우선 이 모델은 순환 신경망(Recurrent Neural Networks, RNN) 모델을 기반으로 하여, 모델은 크게 인코드(Encoder) 부분과 디코더(Decoder) 부분으로 나눈다. 우선 인코더 부분에서 입력값을 받아 입력값의 정보를 담은 벡터를 만들어낸다. 이후 디코더에서는 이 벡터를 활용해 재귀적으로 출력값을 만들어내는 구조다.
- 다음 그림은 "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" 논문에 나온 전체적인 모델에 대한 그림이다. 

![스크린샷 2025-03-29 오전 9 38 01](https://github.com/user-attachments/assets/79353ff0-e794-4117-84a3-aedf88208d55)

- 그림을 보면 전체적인 구조를 직관적으로 이해할 수 있는데, 우선 아래쪽에 나와있는 박스가 인코더다. 각 순환 신경망의 스텝마다 입력값이 들어가고 있다. 이때 입력값은 하나의 단어가 된다. 그리고 인코더 부분의 전체 순환 신경망의 마지막 부분에서  'c'로 표현된 하나의 벡터값이 나온다. 이 벡터가 인코더 부분의 정보를 요약해 담고 있는 벡터이고, 정확하게는 순환 신경망의 마지막 은닉 상태 벡터값을 사용한다.
- 이제 디코더 부분으로 들어가면 이 벡터를 사용해 새롭게 순환 신경망을 시작한다. 그리고 이 신경망의 각 스텝마다 하나씩 출력값이 나온다. 이때의 출력 역시 하나의 단어가 될 것이다. 그리고 디코더 부분의 그림을 자세히 보면 각 스텝에서의 출력값이 다시 다음 스텝으로 들어가는 구조인데, 정확하게 말하면 각 스텝의 출력값이 다음 스텝의 입력값으로 사용된다.
- 이제 실제 예시에 맞게 구성한 그림을 보며 해당 모델을 좀 더 정확하게 이해해 보자. 다음 그림을 보면 한글 입력인 "안녕, 오랜만이야"라는 입력값을 사용해 "그래 오랜만이야"라는 출력을 뽑는 과정이 나온다. 이를 통해 실제로 만들 모델을 알아보자.

![스크린샷 2025-03-29 오후 1 49 02](https://github.com/user-attachments/assets/e765416e-a9d3-4dd9-864e-1a0706e9cdd6)

- 우선 왼쪽에 파란색으로 돼 있는 부분이 인코더이고 오른쪽에 초록색으로 표시돼 있는 부분이 디코더다. 인코더 부분을 보면 우선 각 신경망의 각 스텝마다 단어가 하나씩 들어가고 있다. 각 단어는 임베딩된 벡터로 바뀐 후 입력값으로 사용된다. 순환 신경망의 경우 구현 시 고정된 문장 길이를 정해야 하는데, 이 그림에서는 인코더와 디코더 모두 4로 지정했다. 하지만 입력값을 생각해보면 '안녕'과 '오랜만이야'라는 두 단어만 존재하기 때문에 나머지 빈 2단어를 패딩으로 채워넣었다. 
- 디코더 부분을 살펴보면 우선 최초 입력값은 `\<START\>`라는 특정 토큰을 사용한다. 이는 문장의 시작을 나타내는 토큰이다. 디코더 역시 해당 단어가 임베딩된 벡터 형태로 입력값으로 들어가고 각 스텝마다 출력이 나온다. 이렇게 나온 출력 단어가 다음 스텝의 입력값으로 사용되는 구조다. 이렇게 반복된 후 최종적으로 `\<END\>`라는 토큰이 나오면 문장의 끝으로 보는 형태로 학습을 진행한다. 
- 예시를 보면 알 수 있듯이 데이터 전처리 과정에서 특정 문장 길이로 자른 후 패딩 처리 및 `\<START\>`와 `\<END\>` 등의 각종 토큰을 넣어야 한다. 따라서 전처리 및 모델 구현 부분을 전체적으로 이해하는 것이 중요하다. 이제 모델을 구현해 보자.

### 모델 구현
- 이번 절에서는 모델을 파이썬 파일(preprocess.py)과 주피터 노트북 파일(Preprocess.ipynb, seq2seq.ipynb)로 구현하겠다. 각 파일을 하나씩 보면 preprocess.ipynb에는 데이터를 불러오고 가공하는 다양한 기능이 들어 있고, Preprocess.ipynb는 사전 구성과 학습에 사용될 데이터로 구성돼 있다. seq2seq.ipynb는 모델 구성과 학습, 평가, 실행 등을 할 수 있는 파일이다. 
- 이제 각 파일을 하나씩 살펴보면서 어떤 식으로 구성돼 있는지 확인해 보자. 먼저 설정값을 지정해 둔 preprocess.py 파일부터 살펴보자.

#### preprocess.py

- preprocess.py 파일의 내용은 다음과 같다. 우선 모듈을 불러온다.

```python
import os
import re
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt
```

- 데이터 처리를 위해 활용하는 모듈들이다. 보다시피 한글 형태소를 활용하기 위한 `konlpy`, 데이터를 불러오기 위한 `pandas`, 운영체제의 기능을 사용하기 위한 `os`, 정규표현식을 사용하기 위한 `re`를 불러온다. 
- 불러올 패키지를 정의했으니 이제 학습에 사용할 데이터를 위한 데이터 처리와 관련해서 몇가지 설정값을 지정한다.

```python
FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25
```

- 정규 표현식에서 사용할 필터와 특별한 토큰인 `PAD`, `SOS`, `END`, `UNK`와 해당 토큰들의 인덱스 값을 지정했다. 
- 특별한 토큰의 의미는 아래와 같다. 
  - `PAD`: 어떤 의미도 없는 패딩 토큰이다.
  - `SOS`: 시작 토큰을 의미한다.
  - `END`: 종료 토큰을 의미한다.
  - `UNK`: 사전에 없는 단어를 의미한다.
- 그리고 필터의 경우 정규 표현식 모듈을 사용해 컨파일한다. 이를 미리 컴파일해두면 패턴을 사용할 때 반복적으로 컴파일하는 데 드는 시간을 줄일 수 있다. 
- 다음으로 `load_data` 함수는 데이터를 판다스를 통해 불러오는 함수다.

```python
def load_data(path):
    # 판다스를 통해서 데이터를 불러온다.
    data_df = pd.read_csv(path, header=0)
    # 질문과 답변 열을 가져와 question과 answer에 넣는다.
    question, answer = list(data_df['Q']), list(data_df['A'])

    return question, answer
```

- 판다스를 통해 데이터를 가져와 데이터프레임 형태로 만든 후 question과 answer를 돌려준다. inputs, outputs에는 question과 answer가 존재한다.
- 다음으로 단어 사전을 만들기 위해서는 데이터를 전처리한 후 단어 리스트로 먼저 만들어야 하는데 이 기능을 수행하는 `data_tokenizer` 함수를 먼저 정의한다.

```python
def data_tokenizer(data):
    # 토크나이징 해서 담을 배열 생성
    words = []
    for sentence in data:
        # FILTERS = "([~.,!?\"':;)(])"
        # 위 필터와 같은 값들을 정규화 표현식을
        # 통해서 모두 "" 으로 변환 해주는 부분이다.
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    # 토그나이징과 정규표현식을 통해 만들어진
    # 값들을 넘겨 준다.
    return [word for word in words if word]
```

- 정규 표현식을 사용해 특수 기호를 모두 제거하고 공백 문자를 기준으로 단어들을 나눠서 전체 데이터의 모든 단어를 포함하는 단어 리스트로 만든다. 
- 다음으로 `prepro_like_morphlized` 함수는 한글 텍스트를 토크나이징하기 위해 형태소로 분리하는 함수다. `KoNLPy`에서 제공하는 `Okt` 형태소 분리기를 사용해 형태소 기준으로 텍스트 데이터를 토크나이징한다. 이 함수에서는 환경설정 파일을 통해 사용할지 사용하지 않을지 선택할 수 있다. 

```python
def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data
```

- 보다시피 형태소로 분류한 데이터를 받아 `morphs` 함수를 통해 토크나이징된 리스트 객체를 받고 이를 공백 문자를 기준으로 문자열로 재구성해서 반환한다. 
- 이제 단어 사전을 만드는 함수를 정의하자.

```python
def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    # 사전을 담을 배열 준비한다.
    vocabulary_list = []
    # 사전을 구성한 후 파일로 저장 진행한다.
    # 그 파일의 존재 유무를 확인한다.
    if not os.path.exists(vocab_path):
        # 이미 생성된 사전 파일이 존재하지 않으므로
        # 데이터를 가지고 만들어야 한다.
        # 그래서 데이터가 존재 하면 사전을 만들기 위해서
        # 데이터 파일의 존재 유무를 확인한다.
        if (os.path.exists(path)):
            # 데이터가 존재하니 판단스를 통해서
            # 데이터를 불러오자
            data_df = pd.read_csv(path, encoding='utf-8')
            # 판다스의 데이터 프레임을 통해서
            # 질문과 답에 대한 열을 가져 온다.
            question, answer = list(data_df['Q']), list(data_df['A'])
            if tokenize_as_morph:  # 형태소에 따른 토크나이져 처리
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
            data = []
            # 질문과 답변을 extend을
            # 통해서 구조가 없는 배열로 만든다.
            data.extend(question)
            data.extend(answer)
            # 토큰나이져 처리 하는 부분이다.
            words = data_tokenizer(data)
            # 공통적인 단어에 대해서는 모두
            # 필요 없으므로 한개로 만들어 주기 위해서
            # set해주고 이것들을 리스트로 만들어 준다.
            words = list(set(words))
            # 데이터 없는 내용중에 MARKER를 사전에
            # 추가 하기 위해서 아래와 같이 처리 한다.
            # 아래는 MARKER 값이며 리스트의 첫번째 부터
            # 순서대로 넣기 위해서 인덱스 0에 추가한다.
            # PAD = "<PADDING>"
            # STD = "<START>"
            # END = "<END>"
            # UNK = "<UNKNWON>"
            words[:0] = MARKER
        # 사전을 리스트로 만들었으니 이 내용을
        # 사전 파일을 만들어 넣는다.
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')

    # 사전 파일이 존재하면 여기에서
    # 그 파일을 불러서 배열에 넣어 준다.
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    # 배열에 내용을 키와 값이 있는
    # 딕셔너리 구조로 만든다.
    char2idx, idx2char = make_vocabulary(vocabulary_list)
    # 두가지 형태의 키와 값이 있는 형태를 리턴한다.
    # (예) 단어: 인덱스 , 인덱스: 단어)
    return char2idx, idx2char, len(char2idx)
```
- 단어 사전의 경우 우선적으로 경로에 단어 사전 파일이 있다면 불러와서 사용한다. 만약 없다면 새로 만드는 구조인데, 단어 사전 파일이 없다면 데이터를 불러와서 앞서 정의한 함수를 이용해 데이터를 토크나이징해서 단어 리스트로 만든다. 그 후 파이썬 집합(set) 데이터 타입을 사용해 중복을 제거한 후 단어 리스트로 만든다. 또한 `MARKER`로 사전에 정의한 특정 토큰들을 단어 리스트 앞에 추가한 후 마지막으로 이 리스트를 지정한 경로에 저장한다. 
- 이후에 지정한 경로에 파일이 존재하며, 만약 다시 `load_vocabulary`를 호출한다면 지정한 경로에서 단어 리스트를 불러온 후 `make_vocabulary` 함수의 결과로 `word2idx`, `idx2word`라는 두 개의 값을 얻는데, 각각 단어에 대한 인덱스와 인덱스에 대한 단어를 가진 딕셔너리 데이터에 해당한다. 이 두 값과 단어의 개수를 최종적으로 리턴하면 함수가 끝난다. 
- 그럼 이번에는 `make_vocabulary` 함수를 살펴보자.

```python
def make_vocabulary(vocabulary_list):
    # 리스트를 키가 단어이고 값이 인덱스인
    # 딕셔너리를 만든다.
    char2idx = {char: idx for idx, char in enumerate(vocabulary_list)}
    # 리스트를 키가 인덱스이고 값이 단어인
    # 딕셔너리를 만든다.
    idx2char = {idx: char for idx, char in enumerate(vocabulary_list)}
    # 두개의 딕셔너리를 넘겨 준다.
    return char2idx, idx2char
```

- 함수를 보면 단어 리스트를 인자로 받는ㄴ데 이 리스트를 사용해 두 개의 딕셔너리를 만든다. 하나는 단어에 대한 인덱스를 나타내고, 나머지는 인덱스에 대한 단어를 나타내도록 만든 후 이 두 값을 리턴한다.
- 마지막 줄의 사전 불러오기 함수를 호출하면 각각의 `word2idx`, `idx2word`, `vocab_size`에 앞에서 설명한 값이 들어가고 필요할 때마다 사용한다. 
- 이해를 돕기 위해 예제를 통해 `vocabulary_list`가 어떻게 변하는지 설명하겠다. 만약 `vocabulary_list`에 \[안녕, 너는, 누구야\]가 들어있다고 해보자. `word2idx`에서는 `key`가 '안녕', '너는', '누구야'가 되고 `value`가 0, 1, 2가 되어 {'안녕': 0, '너는': 1, '누구야': 2}가 된다. 반대로 `idx2word`는 `key`가 0, 1, 2가 되고 `value`는 '안녕, '너는', '누구야'가 되어 {0: '안녕', 1: '너는', 2: '누구야'}가 된다.
- 이제 불러온 데이터를 대상으로 인코더 부분과 디코더 부분에 대해 각각 전처리해야 한다. 우선 인코더에 적용될 입력값을 만드는 전처리 함수를 확인해 보자.

```python
def enc_processing(value, dictionary, tokenize_as_morph=False):
    # 인덱스 값들을 가지고 있는
    # 배열이다.(누적된다.)
    sequences_input_index = []
    # 하나의 인코딩 되는 문장의
    # 길이를 가지고 있다.(누적된다.)
    sequences_length = []
    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    # 한줄씩 불어온다.
    for sequence in value:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규화를 사용하여 필터에 들어 있는
        # 값들을 "" 으로 치환 한다.
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 하나의 문장을 인코딩 할때
        # 가지고 있기 위한 배열이다.
        sequence_index = []
        # 문장을 스페이스 단위로
        # 자르고 있다.
        for word in sequence.split():
            # 잘려진 단어들이 딕셔너리에 존재 하는지 보고
            # 그 값을 가져와 sequence_index에 추가한다.
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            # 잘려진 단어가 딕셔너리에 존재 하지 않는
            # 경우 이므로 UNK(2)를 넣어 준다.
            else:
                sequence_index.extend([dictionary[UNK]])
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        # 하나의 문장에 길이를 넣어주고 있다.
        sequences_length.append(len(sequence_index))
        # max_sequence_length보다 문장 길이가
        # 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을
        # sequences_input_index에 넣어 준다.
        sequences_input_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 텐서플로우 dataset에 넣어 주기 위한
    # 사전 작업이다.
    # 넘파이 배열에 인덱스화된 배열과
    # 그 길이를 넘겨준다.
    return np.asarray(sequences_input_index), sequences_length
```
- 함수를 보면 우선 2개의 인자를 받는데, 하나는 전처리할 데이터이고, 나머지 하나는 단어 사전이다. 이렇게 받은 입력 데이터를 대상으로 전처리를 진행하는데, 단순히 띄어쓰기를 기준으로 토크나이징한다.
- 전체적인 전처리 과정을 설명하면 우선 정규 표현식 라이브러리를 이용해 특수문자를 모두 제거한다. 다음으로 각 단어를 단어 사전을 이용해 단어 인덱스로 바꾸는데, 만약 어떤 단어가 단어 사전에 포함돼 있지 않다면 `UNK`토큰을 넣는다(참고로 앞에서 `UNK` 토큰의 인덱스 값은 3으로 설정했다). 이렇게 모든 단어를 인덱스로 바꾸고 나면 모델에 적용할 최대 길이보다 긴 문장의 경우 잘라야 한다. 만약 최대 길이보다 짧은 문장인 경우에는 문장의 뒷부분에 패딩 값을 넣는다. 최대길이가 5라고 가정하고 다음 예시를 참조하자.
  - 인코더 최대 길이보다 긴 경우: "안녕 우리 너무 오랜만에 만난거 같다."
  - 인코더 최대 길이보다 긴 경우 입력값: "안녕, 우리, 너무, 오랜만에, 만난거"
- 위와 같이 최대 길이보다 긴 경우 마지막 단어인 "같다."가 생략된 입력값이 만들어진다. 
  - 인코더 최대 길이보다 짧은 경우: "안녕"
  - 인코더 최대 길이보다 짧은 입력값: "안녕,\<PAD\>,\<PAD\>,\<PAD\>"
- 위와 같이 최대 길이보다 짧은 단어는 최대 길이만큼 모두 패드로 채워진다.
- 함수의 리턴값을 보면 2개의 값이 반환되는 것을 확인할 수 있는데, 하나는 앞서 전처리한 데이터이고 나머지 하나는 패딩 처리하기 전의 각 문장의 실제 길이를 담고 있는 리스트다. 이렇게 두 개의 값을 리턴하면서 함수가 끝난다.
- 이제 디코더 부분에 필요한 전처리 함수를 만들면 된다. 인코더 부분과는 다르게 디코더에는 두 가지 전처리 함수가 사용된다. 디코더의 입력으로 사용될 입력값을 만드는 전처리 함수와 디코더의 결과로 학습을 위해 필요한 라벨인 타깃값을 만드는 전처리 함수다. 예를 들면, "그래 오랜만이야"라는 문장을 전처리하면 다음과 같이 두 개의 값을 만들어야 한다. 
  - 디코더 입력값: "\<SOS\>,그래, 오랜만이야.\<PAD\>"
  - 디코더 타깃값: "그래, 오랜만이야.\<END\>,\<PAD\>"
- 위와 같이 입력값으로 시작 토큰이 앞에 들어가 있고 타깃값은 문장 끝에 종료 토큰이 들어가 있어야 한다. 그리고 예시의 단어는 실제로는 각 단어의 인덱스 값으로 만든다. 
- 디코더의 입력값을 만드는 함수를 살펴보자.

```python
def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    # 인덱스 값들을 가지고 있는
    # 배열이다.(누적된다)
    sequences_output_index = []
    # 하나의 디코딩 입력 되는 문장의
    # 길이를 가지고 있다.(누적된다)
    sequences_length = []
    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    # 한줄씩 불어온다.
    for sequence in value:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규화를 사용하여 필터에 들어 있는
        # 값들을 "" 으로 치환 한다.
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 하나의 문장을 디코딩 할때 가지고
        # 있기 위한 배열이다.
        sequence_index = []
        # 디코딩 입력의 처음에는 START가 와야 하므로
        # 그 값을 넣어 주고 시작한다.
        # 문장에서 스페이스 단위별로 단어를 가져와서 딕셔너리의
        # 값인 인덱스를 넣어 준다.
        sequence_index = [dictionary[STD]] + [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        # 하나의 문장에 길이를 넣어주고 있다.
        sequences_length.append(len(sequence_index))
        # max_sequence_length보다 문장 길이가
        # 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을
        # sequences_output_index 넣어 준다.
        sequences_output_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 텐서플로우 dataset에 넣어 주기 위한
    # 사전 작업이다.
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_output_index), sequences_length
```

- 함수의 구조는 전체적으로 인코더의 입력값을 만드는 전처리 함수와 동일하다. 한 가지 다른점은 각 문장의 처음에 시작 토큰을 넣어준다는 점이다. 디코더 역시 데이터와 단어 사전을 인자로 받고 전처리한 데이터와 각 데이터 문장의 실제 길이의 리스트를 리턴한다. 
- 디코더의 타깃값을 만드는 전처리 함수도 이와 거의 유사하다.

```python
def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    # 인덱스 값들을 가지고 있는
    # 배열이다.(누적된다)
    sequences_target_index = []
    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    # 한줄씩 불어온다.
    for sequence in value:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규화를 사용하여 필터에 들어 있는
        # 값들을 "" 으로 치환 한다.
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 문장에서 스페이스 단위별로 단어를 가져와서
        # 딕셔너리의 값인 인덱스를 넣어 준다.
        # 디코딩 출력의 마지막에 END를 넣어 준다.
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        # 그리고 END 토큰을 넣어 준다
        if len(sequence_index) >= MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE - 1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]
        # max_sequence_length보다 문장 길이가
        # 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을
        # sequences_target_index에 넣어 준다.
        sequences_target_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 텐서플로우 dataset에 넣어 주기 위한 사전 작업이다.
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_target_index)
```

- 위의 디코더의 입력값을 만드는 함수와의 차이점은 문장이 시작하는 부분에 토큰을 넣지 않고 마지막에 종료 토큰을 넣는다는 점이다. 그리고 리턴값이 하나만 있는데, 실제 길이를 담고 있는 리스트의 경우 여기서는 따로 만들지 않았다.

#### preprocess.ipynb

- `preprocess.ipynb` 파일의 내용은 다음과 같다. 이 파일에서는 앞서 구현한 `preprocess.py`의 함수를 이용해 학습 데이터를 준비한다.

```python
from preprocess import *

PATH = 'data_in/ChatBotData.csv_short'
VOCAB_PATH = 'data_in/vocabulary.txt'
```

- 먼저 `preprocess` 모듈에서 모든 함수를 불러오고 학습할 데이터 경로와 저장할 단어사전의 경로를 선언한다. 

```python
inputs, outputs = load_data(PATH)
char2idx, idx2char, vocab_size = load_vocabulary(PATH, VOCAB_PATH, tokenize_as_morph=False)
```

- 앞서 구현한 `load_data` 함수로 학습할 데이터를 불러온다. 그리고 `load_vocabulary` 함수로 단어 사전을 `char2idx`, `idx2char`로 만든다. `tokenize_as_morph` 파라미터를 통해 문장 토크나이즈를 띄어쓰기 단위로 할지 형태소 단위로 할지 결정한다. `tokenize_as_morph`를 False로 설정하면 띄어쓰기 단위로 토크나이즈 한다. 

```python
index_inputs, input_seq_len = enc_processing(inputs, char2idx, tokenize_as_morph=False)
index_outputs, output_seq_len = dec_output_processing(outputs, char2idx, tokenize_as_morph=False)
index_targets = dec_target_processing(outputs, char2idx, tokenize_as_morph=False)
```

- 이렇게 단어 사전까지 만들면 `enc_processing`과 `dec_output_processing`, `dec_target_processing` 함수를 통해 모델에 학습할 인덱스 데이터를 구성한다. 

```python
data_configs = {}
data_configs['char2idx'] = char2idx
data_configs['idx2char'] = idx2char
data_configs['vocab_size'] = vocab_size
data_configs['pad_symbol'] = PAD
data_configs['std_symbol'] = STD
data_configs['end_symbol'] = END
data_configs['unk_symbol'] = UNK
```

- 인덱스 데이터를 모두 구성하고 나면 모델 학습할 때와 모델 추론에 활용하기 위한 단어 사전을 저장할 수 있도록 구성한다. 여기서는 단어 사전과 특별한 토큰들을 각각 정의해서 딕셔너리 객체에 저장한다.

```python
DATA_IN_PATH = './data_in/'
TRAIN_INPUTS = 'train_inputs.npy'
TRAIN_OUTPUTS = 'train_outputs.npy'
TRAIN_TARGETS = 'train_targets.npy'
DATA_CONFIGS = 'data_configs.json'

np.save(open(DATA_IN_PATH + TRAIN_INPUTS, 'wb'), index_inputs)
np.save(open(DATA_IN_PATH + TRAIN_OUTPUTS , 'wb'), index_outputs)
np.save(open(DATA_IN_PATH + TRAIN_TARGETS , 'wb'), index_targets)

json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'))
```

- 각 인덱스 데이터와 단어사전을 구성한 딕셔너리 객체를 `numpy`와 `json` 형식으로 저장한다. 
- 이렇게 하면 모델을 학습할 준비를 마치게 된다. 이제 본격적으로 `seq2seq` 모델을 학습해 보자. 

#### seq2seq.ipynb

- 그럼 모델에 대해 알아보자. 우선 모델을 구현하기 위한 모듈을 불러오자.

```python
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from preprocess import *
```

- 모델 구현을 진행하는 데는 텐서플로와 넘파이를 주로 활용한다. 운영체제의 기능을 사용하기 위한 `os`, 빠른 학습 중지와 모델 체크포인트를 위한 케라스 API를 사용하기 위해 불러오고 있다. 
- 학습 시각화를 위한 시각화 함수를 만들어 보자.

```python
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
```

- 에폭당 정확도와 손실 값을 `matplotlib`을 통해 시각화하는 함수를 만들었다. 이 함수를 통해 직관적으로 학습 상태를 파악할 수 있다.
- 학습 데이터 경로를 정의하고 코드 작성의 효율성을 높여보자.

```python
DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'
TRAIN_INPUTS = 'train_inputs.npy'
TRAIN_OUTPUTS = 'train_outputs.npy'
TRAIN_TARGETS = 'train_targets.npy'
DATA_CONFIGS = 'data_configs.json'
```

- `process.ipynb`에서 만든 npy 데이터와 입력에 필요한 파일이 존재하는 `data_in`, 모델 결과를 저장하는 `data_out`을 선언했다.
- 다음으로 책 전체에서 사용되는 랜덤 시드값을 선언한다.

```python
SEED_NUM = 1234
tf.random.set_seed(SEED_NUM)
```

- 미리 전처리된 학습에 필요한 데이터와 설정값을 불러오자.

```python
index_inputs = np.load(open(DATA_IN_PATH + TRAIN_INPUTS, 'rb'))
index_outputs = np.load(open(DATA_IN_PATH + TRAIN_OUTPUTS , 'rb'))
index_targets = np.load(open(DATA_IN_PATH + TRAIN_TARGETS , 'rb'))
prepro_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))
```

- 인코더의 입력, 디코더의 입력, 디코더의 타깃값을 얻기 위해 앞서 전처리 작업을 하고 저장한 `numpy`와 `json` 파일을 `np.load`와 `json.load`를 통해 불러왔다.
- 앞에서도 설명했지만 인코더의 입력, 디코더의 입력, 디코더의 타깃값을 가져왔으므로 인코더는 최대 길이만큼 \<PAD\>가 붙고, 디코더 입력의 앞에는 \<SOS\>가, 디코더 타깃값 끝에는 \<END\>가 붙은 형태로 만들어졌다는 점을 한번 생각하고 다음으로 진행하자.
- 함수를 통과한 값들이 예상한 크기와 같은지 확인해 보자.

```python
# Show length
print(len(index_inputs),  len(index_outputs), len(index_targets))
```

```
20 20 20
```

- 배열의 크기를 확인하는 함수를 통해 배열의 크기를 확인했다. 만약 앞에서 만들어 둔 `enc_processing`, `dec_output_processing`, `dec_target_processing` 함수를 통해 데이터의 내용을 확인하고 싶다면 `index_inputs`, `index_outputs`, `index_targets`를 출력해 보자.
- 모델을 구성하는 데 필요한 값을 선언해 보자.

```python
MODEL_NAME = 'seq2seq_kor'
BATCH_SIZE = 2
MAX_SEQUENCE = 25
EPOCH = 30
UNITS = 1024
EMBEDDING_DIM = 256
VALIDATION_SPLIT = 0.1 

char2idx = prepro_configs['char2idx']
idx2char = prepro_configs['idx2char']
std_index = prepro_configs['std_symbol']
end_index = prepro_configs['end_symbol']
vocab_size = prepro_configs['vocab_size']
```
- 배치 크기(BATCH_SIZE), 에폭 횟수(EPOCH), 순환 신경망의 결과 차원(UNITS), 임베딩 차원(EMBEDDING_DIM)과 전체 데이터셋 크기에서 평가셋의 크기 비율(VALIDATION_SPLIT) 등을 선언하고 사용할 것이다. 에폭(EPOCH)는 전체 학습 데이터를 전체 순회하는 것이 한 번, 즉 1회다. 전체 데이터셋 크기에서 평가셋의 크기 비율(VALIDATION_SPLIT)은 데이터의 전체 크기대비 평가셋의 비율을 의미한다. 예를 들어, 전체 데이터셋이 100개의 셋으로 구성돼 있다고 했을 때 0.1은 10개를 의미한다. 이어서 `preprocess.ipynb`에서 만들어 둔, 토큰을 인덱스로 만드는 함수와 인덱스를 토큰으로 변환하는 함수, 특수 토큰인 시작 토큰과 끝 토큰 값, 사전의 크기를 차례로 불러왔다. `preprocess.ipynb`에서 만들어 둔 값들은 모델을 만들 때 유용하게 쓰이는 값이다. 이 뒤에 모델 구현을 진행하면서 미리 만들어 둔 값들이 어디에 쓰이는지 확인하는 것도 공부에 도움이 될 것이다. 
- 앞에서 설명했듯이 모델은 시퀀스 투 시퀀스 모델을 기반으로 만들 것이다. 해당 모델의 중간에 사용되는 신경망으로는 순환 신경망을 사용하는데, 다양한 종류의 순환 신경망 중에서 여기에서는 조경현 교수님이 2014년에 발표한 `GRU(Gated Recurrent Unit)` 모델을 사용하겠다.
- 시퀀스 투 시퀀스 모델의 인코더부터 실펴보자.

```python
lass Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size 
        self.embedding_dim = embedding_dim          
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self, inp):
        return tf.zeros((tf.shape(inp)[0], self.enc_units))
```