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



