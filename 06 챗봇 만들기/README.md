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
class Encoder(tf.keras.layers.Layer):
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

- Encoder 클래스는 `Layer`를 상속받고 있으며, `__init__` 함수부터 설명하겠다. 임베딩 룩업 테이블 결과 `GRU`를 구성하기 위한 인자를 입력으로 받는다. 인자는 배치 크기(batch_sz), 순환 신경망의 결과 차원(enc_units), 사전 크기(vocab_size), 임베딩 차원(embedding_dim)이다. 함수로는 `tf.keras.layers.Embedding`과 `tf.keras.layers.GRU`가 있다. `tf.keras.layers.Embedding` 함수는 사전에 포함된 각 단어를 `self.embedding_dim` 차원의 임베딩 벡터로 만든다. `tf.keras.layers.GRU` 함수는 `GRU` 신경망을 만드는 부분이다. 인자로 전달되는 `self.enc_units`는 `GRU`의 결과 차원의 크기라고 이야기 했다. `return_sequences`는 각 시퀀스마다 출력을 반환할지 여부를 결정하는 값이며, 해당 모델에서는 각각의 시퀀스마다 출력을 반환한다. `return_state`는 마지막 상태 값의 반환 여부이며, 해당 모델은 상태값을 반환한다. 마지막 `recurrent_initializer`에는 초기값을 무엇으로 할 것인지 선언할 수 있으며, `glorot_uniform`은 `Glorot` 초기화 `Xavier` 초기화라고 불리는 초기화 방법으로, 이전 노드와 다음 노드의 개수에 의존하는 방법이다. `Uniform` 분포를 따르는 방법과 `Normal` 분포를 따르는 두 가지 방법이 사용되는데, 여기서는 `Glorot Uniform` 방법을 이용한 초기화 방법을 선택했다.
- `call` 함수는 입력값 `x`와 은닉 상태 `hidden`을 받는다. `__init__` 함수에서 만들어 놓은 `embedding` 함수를 통해 `x` 값을 임베딩 벡터로 만든다. 그리고 `gru` 함수에 임베딩 벡터와 순환 신경망의 초기화 상태로 인자로 받은 은닉 상태를 전달하고, 결과값으로 시퀀스의 출력값과 마지막 상태값을 리턴한다. `tf.keras.layers`의 함수들은 고수준 API라서 이처럼 사용하기가 간편하다. 
- 마지막으로 `initialize_hidden_state` 함수는 배치 크기를 받아 순환 신경망에 초기에 사용될 크기의 은닉 상태를 만드는 역할을 한다. 
- 시퀀스 투 시퀀스 모델링에서 설명한 인코더 디코더 구조는 시퀀스 투 시퀀스의 문제점을 보완하기 위해 나온 개념이며, 기존의 시퀀스 투 시퀀스는 문장이 길어질수록 더 많은 정보를 고정된 길이에 담아야 하므로 정보의 손실이 있다는 점이 큰 문제로 지적됐다. 추가로 순환 신경망 특유의 문제인 장기 의존성 문제가 발생할 수 있는 부분 또한 문제점으로 지적됐다. 이러한 기존의 문제를 어텐션 방법을 통해 보완했다. 

![스크린샷 2025-03-30 오후 12 01 49](https://github.com/user-attachments/assets/c7bcb705-bdcb-4150-abcd-76b849cb5707)

- 기존의 시퀀스 투 시퀀스는 인코더의 고정된 문맥 벡터가 디코더로 전달된다면 어텐션이 추가된 방법은 은닉 상태의 값을 통해 어텐션을 계산하고 디코더의 각 시퀀스 스텝마다 계산된 어텐션을 입력으로 넣는다. 즉, 어텐션도 함께 학습을 진행하게 되며 학습을 통해 디코더의 각 시퀀스 스텝마다 어텐션의 가중치는 다르게 적용된다. 
- 그럼 어텐션 소스코드를 살펴보자. 

```python
 class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
```

- `BahdanauAttention` 클래스의 `__init__` 함수는 출력 벡터의 크기를 인자로 받는다. `tf.keras.layers.Dense` 함수를 통해 출력 크기가 `units` 크기인 `W1`과 `W2`, 출력 크기가 1인 v의 완전 연결 계층을 만든다. 
- `call` 함수의 인자인 `query`는 인코더 순환 신경망의 은닉층의 상태 값이고, `values`는 인코더 재귀 순환망의 결괏값이다. 첫 번째 줄에서 `query`를 `W2`에 행렬곱을 할 수 있는 형태(shape)를 만든다. 두 번째 줄에서 `W1`과 `W2`의 결괏값의 요소를 각각 더하고 하이퍼볼릭 탄젠트 활성함수를 통과한 값을 v에 행렬곱하면 1차원 벡터값이 나온다. 모델 훈련 중 `W1`, `W2`, `V` 가중치들은 학습된다. 소프트맥스 함수를 통과시켜 어텐션 가중치를 얻는데, `attention_weights` 값은 모델이 중요하다고 판단하는 값은 1에 가까워지고, 영향도가 떨어질수록 0에 가까운 값이 된다. `attention_weights` 값을 `value`, 즉 순환신경망 결괏값에 행렬 곲을 하게 되면 1에 가까운 값에 위치한 `value` 값은 커지고 0에 가까운 값에 위치한 `value` 값은 작아진다. 
- 결과적으로 인코더 순환 신경망의 결괏값을 어텐션 방법을 적용해 가중치를 계산해서 가중치가 적용된 새로운 인코더 순환 신경망의 결과값을 만들어내서 디코더에 전달하게 되며, 이때 만들어진 `Attention` 클래스에 포함된 `W1`, `W2`, `V`는 학습을 통해 값들이 최적화되며 기존 시퀀스 투 시퀀스의 문제를 해결하는 방법론이 적용된다.
- 디코더의 소스코드를 살펴보자.

```python
class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size = vocab_size 
        self.embedding_dim = embedding_dim  
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(self.vocab_size)

        self.attention = BahdanauAttention(self.dec_units)
        
    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
            
        x = self.fc(output)
        
        return x, state, attention_weights
```

- Decoder 클래스의 `__init__` 함수는 Encoder 클래스의 `__init__` 함수와 유사하며, 다른 부분만 설명하겠다. 출력 값이 사전 크기인 완전 연결 계층 `fc`를 만들고 `BahdanauAttention` 클래스를 생성한다.
- `call` 함수는 디코더의 입력값 `x`와 인코더의 은닉 상태 값 `hidden`, 인코더의 결과값을 인자로 받는다. 
- `self.attention` 함수를 호출하면 `BahdanauAttention` 클래스의 `call` 함수가 호출되고 앞에서 설명한 값에 따라 어텐션이 계산된 문맥 벡터(context_vector)를 돌려받는다. 디코더의 입력값을 `self.embedding` 함수를 통해 임베딩 벡터를 받고 문맥 벡터와 임베딩 벡터를 결합해 x를 구성하고 디코더 순환 신경망을 통과해 순환 신경망의 결과값(output)을 얻게 되고 이 값을 완전 연결 계층(fully-connected layer)을 통과해서 사전 크기의 벡터 x를 만든다. 각각의 독립적인 클래스 인코더, 디코더, 어텐션을 살펴봤다.
- 이어서 손실 함수와 정확도 측정 함수를 살펴보자.

```python
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

def loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def accuracy(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)
    pred *= mask    
    acc = train_accuracy(real, pred)

    return tf.reduce_mean(acc)
```

- 여기서는 세 가지를 미리 생성하는데, 최적화로 아담을 사용하기 위한 객체(optimizer), 크로스 엔트로피로 손실 값을 측정하기 위한 객체(loss_object), 정확도 측정을 위한 객체(train_accuracy)를 생성한다. 
- `loss` 함수는 인자로 정답과 예측한 값을 받아서 두 개의 값을 비교해서 손실을 계산하며, real 값 중 0인 값 \<PAD\>는 손실 계산에서 빼기 위한 함수다.
- `accuracy` 함수는 `loss` 함수와 비슷하며, 다른 점은 `train_accuracy` 함수를 통해 정확도를 체크한다는 것이다. 
- `loss` 함수와 `accuracy` 함수에 동일하게 등장하는 `mask`를 한번 보자.
- 첫 번째 줄에 등장하는 `tf.mat.logical_not(tf.math.equal(real, 0)`은 정답 real에 포함되는 값 중 0인 것은 \<PAD\>를 의미하는 값이며, 해당 값들은 True(1)가 되고 \<PAD\>를 제외한 나머지 값들은 False(0)가 된다. 치환된 요소들의 값에 `logical_not` 함수를 적용하면 각 요소들의 값은 0에서 1로, 1에서 0으로 변경된다. 이렇게 변경된 값은 `loss_ *= mask`에 요소 간에 곱을 해주면 \<PAD\> 부분들은 loss_ 계산에서 빠진다. 또한 `pred *= mask`를 수행하면 정확도 측정에서 빠진다. 
- 이제 살펴볼 `seq2seq`클래스는 각각 분리돼 있는 각 클래스를 이어주는 메인 클래스로 볼 수 있다.

```python
class seq2seq(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, batch_sz, end_token_idx=2):    
        super(seq2seq, self).__init__()
        self.end_token_idx = end_token_idx
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units, batch_sz) 
        self.decoder = Decoder(vocab_size, embedding_dim, dec_units, batch_sz) 

    def call(self, x):
        inp, tar = x
        
        enc_hidden = self.encoder.initialize_hidden_state(inp)
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        predict_tokens = list()
        for t in range(0, tar.shape[1]):
            dec_input = tf.dtypes.cast(tf.expand_dims(tar[:, t], 1), tf.float32) 
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_tokens.append(tf.dtypes.cast(predictions, tf.float32))   
        return tf.stack(predict_tokens, axis=1)
    
    def inference(self, x):
        inp  = x

        enc_hidden = self.encoder.initialize_hidden_state(inp)
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)

        dec_hidden = enc_hidden
        
        dec_input = tf.expand_dims([char2idx[std_index]], 1)
        
        predict_tokens = list()
        for t in range(0, MAX_SEQUENCE):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_token = tf.argmax(predictions[0])
            
            if predict_token == self.end_token_idx:
                break
            
            predict_tokens.append(predict_token)
            dec_input = tf.dtypes.cast(tf.expand_dims([predict_token], 0), tf.float32)   
            
        return tf.stack(predict_tokens, axis=0).numpy()
```

- `seq2seq` 클래스의 `__init__` 함수는 `Encoder` 클래스를 생성할 떄 필요한 값과 `Decoder` 클래스를 생성할 떄 필요한 인자값을 받는다. 
- `call` 함수는 인코더의 입력값과 디코더의 입력값을 `x`를 통해 받는다. `self.encoder`를 통해 인코더 결괏값과 인코더 은닉 상태값을 만든다. 디코더는 시퀀스 최대 길이만큼 반복하면서 디코더의 출력값을 만들어낸다.
- 시퀀스마다 나온 결과값을 리스트(`predict_tokens`)에 넣어 손실 계산 또는 정확도를 계산하는 용도로 사용된다.
- `inference` 함수는 사용자의 입력에 대한 모델의 결괏값을 확인하기 위해 테스트 목적으로 만들어진 함수이며, 하나의 배치만 동작하도록 돼 있으며, \<END\> 토큰을 만나면 반복문을 멈춘다. 전체적인 함수 구조는 `call` 함수와 유사하다.
- `seq2seq`를 만들어 보자.

```python
model = seq2seq(vocab_size, EMBEDDING_DIM, UNITS, UNITS, BATCH_SIZE, char2idx[end_index])
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=[accuracy])
```

- `seq2seq` 객체를 생성한다. 그리고 `compile` 함수를 통해 학습 방식 설정을 한다. 설정은 손실 함수, 최적화 함수, 성능 측정 함수를 설정한다.
- 학습을 진행해 보자.

```python
PATH = DATA_OUT_PATH + MODEL_NAME
if not(os.path.isdir(PATH)):
        os.makedirs(os.path.join(PATH))
        
checkpoint_path = DATA_OUT_PATH + MODEL_NAME + '/weights.h5'
    
cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)

history = model.fit([index_inputs, index_outputs], index_targets,
                    batch_size=BATCH_SIZE, epochs=EPOCH,
                    validation_split=VALIDATION_SPLIT, callbacks=[earlystop_callback, cp_callback])
```

- 체크포인트가 저장될 폴더를 만든다. 그리고 두 개의 함수를 정의한다. 첫째, 모델 체크포인트를 어떻게 저장할지에 대한 정책을 정의한 `ModelCheckpoint` 함수와 학습을 조기 종료할 정책을 정의한 `EarlyStopping`을 선언한다. 그리고 `model.fit`을 통해 학습을 진행한다. `fit` 함수의 첫 번째 배열에 들어가는 것은 인코더의 입력과 디코더의 입력이며, 두 번째 인자인 `index_targets`는 정답이다. 여기에 정의된 두 함수인 `ModelCheckpoint`와 `EarlyStopping`을 `callbacks` 인자에 넣으면 정책에 따라 자동으로 구동된다. 
- 학습과 평가 정확도를 시각화한 그래프를 확인하자.

```python
plot_graphs(history, 'accuracy')
```

![스크린샷 2025-03-30 오후 4 11 20](https://github.com/user-attachments/assets/22c5991c-41a5-41c8-80ef-722c54a3f71a)

- 20부터 조금씩 올라가는 것을 볼 수 있는데 `EarlyStopping`의 `patience=10`값을 높이고 낮추면서 시각화 그래프가 어떻게 변화하는지 확인하자. 이를 확인하는 이유는 `patience`의 값이 오버피팅에 어떠한 영향을 미치는지 경험적으로 이해하기 위해서다.

```python
plot_graphs(history, 'loss')
```

![스크린샷 2025-03-30 오후 4 11 42](https://github.com/user-attachments/assets/3f0e1b57-5ce2-4a95-9454-7dbc24ed0632)

- 저장된 모델을 로드하는 방법을 보자.

```python
SAVE_FILE_NM = "weights.h5"
model.load_weights(os.path.join(DATA_OUT_PATH, MODEL_NAME, SAVE_FILE_NM))
```

- 방금 학습을 진행한 가중치를 불러올 수도 있고 다른 곳에서 학습한 가중치를 불러와 사용할 수도 있다. 물론 순차적으로 진행했다면 해당 소스가 없어도 최신 학습 가중치를 불러올 수 있지만 저장해둔 값을 불러와 사용하고 싶다면 이 부분을 적용하면 된다. 그리고 `SAVE_FILE_NM`은 저장된 값을 확인해 적용해야 한다(이 부분은 조심하자).
- 학습이 잘 됐는지 확인해 보자.

```python
query = "남자친구 승진 선물로 뭐가 좋을까?"

test_index_inputs, _ = enc_processing([query], char2idx)    
predict_tokens = model.inference(test_index_inputs)
print(predict_tokens)

print(' '.join([idx2char[str(t)] for t in predict_tokens]))
```

```
[83 79 98 97 21 56]
평소에 필요했던 게 좋을 것 생각해보세요
```

- `seq2seq` 모델 안에 있는 `inference` 함수를 통해 결과를 확인했다. 
- 이렇게 해서 시퀀스 투 시퀀스 기본 모델에 어텐션(`Attention`) 기법을 추가한 모델을 만들어서 챗봇 기능을 구현해 봤다. 
- 데이터가 전체적으로 연애에 관련된 데이터이므로 다양한 답변을 요구하거나 더 높은 정확도의 답변을 원한다면 충분한 양의 데이터와 다양한 데이터로 학습을 진행해서 결과를 보면 좋을 듯하다.
- 다음으로는 자연어 처리의 다양한 곳에서 활용되고 텍스트 생성 및 대화 모델 등에서 좋은 성능을 보이고 있는 모델인 구글의 트랜스포머 모델을 사용해 챗봇을 만들어 보겠다. 여기서 구현했던 시퀀스 투 시퀀스 모델과 비교하면서 구현 방법이나 모델의 차이점에 대해서도 생각해 보자.


## 트랜스포머 모델

- 앞서 다룬 챗봇은 순환 신경망을 기반으로 한 시퀀스 투 시퀀스 모델을 사용해서 만들었다. 이번 절에서는 시퀀스 투 시퀀스 계열 모델 중에서 많은 사람들이 사용하고 성능이 좋은 최신 모델인 트랜스포머(Transformer) 모델을 만들어 보겠다.
- 트랜스포머란 구글이 2017년에 소개한 논문인 `Attention is all you need`에 나온 모델로서 기존의 시퀀스 투 시퀀스의 인코더 디코더 구조를 가지고 있지만 합성곱 신경망(CNN), 순환 신경망(RNN)을 기반으로 구성된 기존 모델과 다르게 단순히 어텐션 구조만으로 전체 모델을 만들어 어텐션 기법의 중요성을 강조했다. 
- 이 모델 역시 기존의 시퀀스 투 시퀀스와 비슷하게 기계 번역, 문장 생성 등 다양한 분야에 사용되고, 대부분 좋은 성능을 보여준다. 따라서 이번 절에서는 이 모델을 이용해 챗봇을 만들어 보겠다. 먼저 모델에 대해 좀 더 알아보자.

### 모델 소개

- 앞에서 다룬 순환 신경망 모델은 시퀀스 순서에 따른 패턴을 보는 것이 중요했다. 예를 들어, '나는 어제 기분이 좋았어'라는 문장을 시퀀스 투 시퀀스를 거쳐 '기분이 좋다니 저도 좋아요'라고 문장을 생성한다고 해보자. 순환 신경망의 경우 인코더에서 각 스텝을 거쳐 마지막 스텝의 은닉 상태 벡터에 '기분이 좋다'는 문장의 맥락 정보가 반영되어 디코더에 응답 문장을 생성할 수 있다. 
- 이러한 순환 신경망 구조를 통해 맥락 정보를 추출하는 것은 보통의 경우에는 좋은 성능을 보여왔다. 하지만 단순히 하나의 벡터에 인코더 부분에 해당하는 문장에 대한 모든 정보를 담고 있어 문장 안의 개별 단어와의 관계를 확인하기는 어렵다. 또한 문장 길이가 길수록 모든 정보를 하나의 벡터에 포함하기에는 부족하다는 단점이 있다. 예를 들어, 다음과 같은 문장이 인코더에 입력으로 들어간다고 생각해 보자.

```
이러저러한 이유로 엄마가 산타에게 키스하는 그런 장면을 목격했던 것도 아니었지만 어린 나이에 크리스마스에만 일하는 그 영감의 존재를 이상하게 생각했던 매우 똑똑한 아이였던 내가, 어쩐일인지 우주인이니, 미래에서 온 사람이니, 유령이니, 요괴니, 초능력이니, 악의 조직이니 하는 것들과 싸우는 애니메이션, 특촬물, 만화의 히어로들이 이 세상에 존재하지 않는다는 사실을 깨달은 것은 상당히 시간이 지난 뒤의 일이었다. 

- <스즈미야 하루히의 우울> 콘의 독백 중에서
```

- 위의 문장을 순환 신경망 계열 시퀀스 투 시퀀스 모델에 입력한다면 스텝마다 각 단어가 입력되고 은닉 상태 벡터에 반영될 것이다. 앞부분에서 나온 '엄마가'라는 정보 역시 은닉 상태 스테이트에 반영된다. 그 이후 계속해서 나오는 단어들이 입력되면서 은닉 상태 벡터값에 누적될 텐데 문장의 마지막 부분인 '히어로들이'라는ㄴ 문장이 나올 떄면 앞서 반영된 '엄마가'라는 정보는 많이 손실된 상태일 것이다. 따라서 이렇게 문장이 긴 경우에는 모든 단어의 정보가 잘 반영된다고 보기는 어렵다. 그뿐만 아니라 각 단어 간의 유의미한 관계를 잡아내는 것 또한 어려울 것이다. 이러한 순환 신경망 기반의 시퀀스 투 시퀀스 모델의 한계를 지적하고 극복한 모델이 트랜스포머다.
- 기본적으로 트랜스포머 모델은 앞서 순환 신경망 시퀀스 투 시퀀스 모델과 같이 인코더와 디코더로 구성되며 인코더에 입력한 문장에 대한 정보와 디코더에 입력한 문장 정보를 조합해서 디코더 문장 다음에 나올 단어에 대해 생성하는 방법이다. 시퀀스 투 시퀀스와 다른 점은 순환 신경망을 활용하지 않고 `셀프 어텐션 기법`을 사용해 문장에 대한 정보를 추출한다는 점이다.
- 전체 구조를 이해하기 위해서는 먼저 모델에 사용되는 셀프 어텐션을 이해해야 한다. 먼저 셀프 어텐션이라는 정보가 어떻게 생성되고 모델이 어떠한 방식으로 추론하는지 자세히 알아보자.

### 셀프 어텐션

- 셀프 어텐션(`Self-Attention`)이란 문장에서 각 단어끼리 얼마나 관계가 있는지를 계산해서 반영하는 방법이다. 즉, 셀프 어텐션을 이용하면 문장 안에서 단어들 간의 관계를 측정할 수 있다. 이때 각 단어를 기준으로 다른 단어들과 관계 값을 계산한다. 이 값을 어텐션 스코어(`attention score`)라 부른다. 관계도가 큰 단어 간의 어텐션 점수는 높게 나올 것이다. 예를 들어, 다음과 같이 "딥러닝 자연어 처리 아주 좋아요"라는 문장이 주어졌다고 하자.

![스크린샷 2025-03-30 오후 4 35 58](https://github.com/user-attachments/assets/5cf07796-965a-46ec-bb7e-62267d609be0)

- 위와 같이 문장이 주어졌을 때 우선 "딥러닝" 이라는 단어를 기반으로 나머지 단어와의 관계를 측정한다. 그림을 보면 "자연어"라는 단어가 가장 점수가 높게 나왔으며 "좋아요"라는 단어의 경우도 연관도가 가장 낮아 점수가 가장 작게 나왔다. 이렇게 "딥러닝"이라는 단어에 대한 각 단어의 어텐션 스코어를 구했다면 이제 다음 단어인 '자연어'라는 단어에 대해서도 스코어를 구하고, 나머지 모든 단어에 대해 각각 구해야 한다. 이때 각 단어 간의 관계를 측정한 값을 어텐션 스코어라 하고, 이 어텐션 스코어 값을 하나의 테이블로 만든 것을 어텐션 맵이라 부른다. 이제 이 어텐션 맵을 활용해 문장을 서로의 관계를 반영한 값으로 바꿔야 한다.
- 여기까지가 셀프 어텐션에 대한 대략적인 설명이고, 이제 구체적으로 위의 예시 문장을 통해 셀프 어텐션에 대해 알아보자, 우선 위의 문장이 모델에 적용될 때는 각 단어가 임베딩된 벡터 형태로 입력될 것이다. 즉, 다음 그림과 같이 문장이 각 단어 벡터의 모임으로 구성될 것이다.

![스크린샷 2025-03-30 오후 4 40 59](https://github.com/user-attachments/assets/94ad64ec-97c8-4fd9-b11b-f1ead0cba9a6)

- 이처럼 문장에 대한 정보가 단어 벡터로 구성돼 있다고 하면 이제 각 단어 간의 관계를 나타내는 어텐션 스코어를 구해야 한다. 어텐션 스코어를 구하는 방법은 앞서 5장에서 다룬 텍스트 유사도를 구하는 방식과 유사하다. 텍스트의 의미 정보를 벡터로 표현해서 유사도 점수를 계산한다. 앞서 유사도를 구한 방법은 벡터에 대한 맨해튼 거리와 같은 유사도 공식을 활용해 구하는 방법과 `Dense` 층을 거쳐 나온 값을 활용하는 방법이 있었다. 트랜스포머 모델에서는 단어 벡터끼리 내적 연산을 함으로써 어텐션 스코어를 구한다. 

![스크린샷 2025-03-30 오후 4 41 08](https://github.com/user-attachments/assets/1d6872a1-ea30-415d-b252-521b4c591718)

- 특정 단어에 대해 다른 단어들과의 어텐션 스코어를 구한 후, 어텐션 스코어가 모여 있는 어텐션 맵에 소프트맥스 함수를 적용한다. 이렇게 하면 이제 어텐션 맵이 특정 단어에 대한 다른 단어와의 연관도 값의 확률로 나타낸다. 이제 이 값을 해당 단어에 반영해야 한다. 스코어가 큰 경우 해당 단어와 관련이 높은 단어이므로 큰 값을 가져야 한다. 따라서 이 확률값과 기존의 단어 벡터를 가중합한다. 가중합이란 각 확률값과 각 단어 벡터를 곱한 후 더하는 연산이다. 이렇게 구한 값을 해당 단어에 대한 벡터값으로 사용하게 된다. 다음 그림은 위의 과정을 도식화한 것이다. 

![스크린샷 2025-03-30 오후 4 46 58](https://github.com/user-attachments/assets/d0ebe963-4708-48c4-a1f3-5819bd639005)

- 그림 6.17은 '딥러닝'이라는 단어에 대해 셀프 어텐션을 하기 위해 가중합을 구하는 과정을 보여준다. 우선 '딥러닝'이라는 단어를 기준으로 '자연어, '처리', '아주', '좋아요'라는 단어에 대한 어텐션 스코어를 구한다. 그리고 이렇게 구한 어텐션 스코어에 대해 소프트맥스를 적용한 후 각 벡터와 곱한 후 전체 벡터를 더해서 '딥러닝'에 대한 문맥 벡터를 구한다. 이러한 방식으로 나머지 단어에 대해서도 동일하게 진행하면 해당 문장에 대한 셀프 어텐션이 끝난다. 
- 셀프 어텐션 방식은 트랜스포머 네트워크 핵심이며, 이 방식을 통해 어텐션 기법이 적용된 문맥 벡터가 생성된다. 이제 트랜스포머 네트워크 모델의 핵심을 이해했다면 본격적으로 챗봇 모델 구현에 대해 알아보자.

### 모델 구현

- 트랜스포머 모델 구현의 경우 큰 틀은 앞서 구현한 순환 신경망 기반 시퀀스 투 시퀀스 모델과 거의 유사하다. 데이터를 불러오는 부분 등 부가적인 부분은 동일하고 모델 부분만 다르게 구성돼 있다. 모델의 경우 큰 틀은 인코더와 디코더로 구성돼 있다. 따라서 입력이 인코더에 들어가면서 셀프 어텐션 기법을 활용해 해당 문장의 정보를 추출하고 이 값을 토대로 디코더에서 출력 문장을 만들어낸다. 다음 그림을 보자.

![스크린샷 2025-03-30 오후 4 52 48](https://github.com/user-attachments/assets/f52bc638-0300-462b-9cdb-02a39003bfeb)

- 이제 챗봇 프로젝트를 본격적으로 구현해 보자. 프로젝트는 앞에서 살펴본 순환 신경망 기반 시퀀스 투 시퀀스 소스코드와 동일하게 구성한다. 기본적으로 이전 장과 동일한 부분에 대해서는 설명을 생략하고 바뀐 부분 위주로 설명하겠다.

#### 데이터 전처리

- 데이터 구성은 앞서 `preprocess.ipynb`를 통해 준비하면 된다. 이번에는 띄어쓰기가 아닌 형태소 단위인 토크나이즈하는 방식으로 전처리한다. 

```python
os.remove(VOCAB_PATH)
```

- 전처리 방식을 다르게 할 때는 `data_in` 폴더에 `vocabulary.txt` 파일이 존재하는지 확인하고, 파일이 존재하면 `rm` 명령어를 활용해 삭제하고 전처리를 진행한다.

```python
index_inputs, input_seq_len = enc_processing(inputs, char2idx, tokenize_as_morph=True)
index_outputs, output_seq_len = dec_output_processing(outputs, char2idx, tokenize_as_morph=True)
index_targets = dec_target_processing(outputs, char2idx, tokenize_as_morph=True)
```

- 앞서 `seq2seq` 모델에서 수행한 전처리와 다른 점은 `enc_processing`, `dec_output_processing`, `dec_target_processing` 함수의 `tokenize_as_morph` 파라미터를 True로 지정한다는 점이다.

#### 모델 구현

- 앞의 챗봇 프로젝트와 마찬가지로 모델은 `transformer.ipynb`에 구현돼 있다. 이제 모델의 각 부분을 하나씩 구현해 보겠다. 우선 앞에서 모델을 나타낸 그림을 통해 구현해야 하는 모듈로 어떤 것이 있는지 확인해 보자.

![스크린샷 2025-03-30 오후 5 01 25](https://github.com/user-attachments/assets/6a533818-7369-49c8-bb5e-597647acb3df)

- 그림에 나온 각 모듈을 정리하면 구현해야 하는 모듈의 목록은 다음과 같다.
  - 멀티 헤드 어텐션(Multi-head attention)
  - 서브시퀀트 마스크 어텐션(Subsequent masked attention)
  - 포지션-와이즈 피드 포워드 네트워크(Position-wise feed forward network)
  - 리지듀얼 커넥션(Residual connection)
- 위와 같은 총 4가지 모듈을 직접 만들고 구현해야 한다. 이러한 모듈을 정의한 후 각 모듈을 사용해 인코더와 디코더 모듈을 만들어 보겠다. 우선 멀티 헤드 어텐션에 대해 알아보자.

#### 멀티 헤드 어텐션

- 앞에서 설명한 셀프 어텐션 방식을 기반으로 구성된 멀티 헤드 어텐션(Multi-head attention)을 알아보자. 멀티 헤드 어텐션이란 간단하게 내적 어텐션 구조가 중첩된 형태다.
- 이 모듈은 앞서 설명한 기본적인 내적 어텐션, 순방향 어텐션 마스크, 멀티 헤드 어텐션의 세개의 부분으로 나눠서 구현하겠다. 우선 내적 어텐션을 알아보자.

##### 스케일 내적 어텐션

- 앞에서 개념적으로 설명했던 셀프 어텐션을 구체적으로 구현하겠다. 기본적인 내적 어텐션의 입력은 세가지로 구성되며, `query(질의)`, `key(키)`, `value(값)`로 들어온다. `query`, `key`, `value`는 주로 사전에서 많이 쓰이는 형태다. 즉, 찾고자 하는 단어가 있고 영어사전을 통해 의미를 찾을 때, 찾고자 하는 단어는 `query`가 되고 사전에 등록된 단어는 `key`가 될 것이다. 그리고 `key`에 해당하는 단어의 의미는 `value`로 볼 수 있다. 이 같은 구성으로 어떻게 셀프 어텐션을 구현하는지 보자.
- 앞에서 다룬 '딥러닝 자연어 처리 아주 좋아요'라는 문장을 다시 예로 들어보자.

![스크린샷 2025-03-30 오후 8 03 03](https://github.com/user-attachments/assets/6b94e0fc-3232-43bb-a7fd-d0358f34d491)

- 앞에서 알아본 어텐션 구조와 동일한 형태다. 특정 단어에 대해 다른 단어와의 관계를 알아보는 것이 어텐션 구조라고 했는데, 이때 특정 단어가 query가 되는 것이다. 그리고 이 단어와 다른 단어들과의 관계를 알아봐야 하는데, 이때 다른 단어가 `key`, `value`가 된다. 연산 과정을 다시 생각해보자. 우선 특정 단어에 대해 다른 모든 단어와 내적해서 어텐션 스코어값을 구했다. 따라서 `query`에 대해 다른 모든 `key`와 내적해서 어텐션 스코어를 구하는 것이다. 이렇게 구한 스코어에 소프트맥스 함수를 적용해 확률값으로 만든 후 다시 다른 모든 단어들, 즉 value에 각각 곱해서 `query`와의 관계가 적용된 `value` 값으로 만든다. 이제 이 값을 모두 더하면 query에 대한 문맥 벡터가 되는 것이다. 
- 좀 더 자세한 예시를 들어보자. 위 그림의 예시 문장인 "딥러닝 자연어 처리 아주 좋아요"에서 '딥러닝'이라는 단어에 대해 컨텍스트 벡터를 구한다고 하자. 여기서 `query`는 '딥러닝'이라는 단어가 될 것이다. 이제 이 `query`를 이용해 문장의 나머지 단어들과의 관계를 통해 컨텍스트 벡터를 구해야 한다. 이때 나머지 단어들이 `key`와 `value`가 될 것이다. 이제 `query`와 `key`와의 어텐션 스코어를 구한다. 즉 '딥러닝'과 다른 단어들 모두와의 어텐션 스코어를 구하는 것이다. 이렇게 구한 어텐션 스코어를 소프트맥스 함수를 통해 확률값으로 만든 후 `value`와 곱해서 가중합을 취하면 컨텍스트 벡터를 구하게 된다. 
- 여기까지는 앞서 알아본 셀프 어텐션의 구조와 동일하다. 하지만 어텐션의 이름이 내적 어텐션(dot product attention)이 아니라 스케일 내적 어텐션(scaled dot product attention)이다. 이는 중간에 크기를 조정하는 과정(scaling)이 추가된 것이다. 해당 모델을 제시한 논문에서 부가적으로 추가한 기법으로 `query`와 `value`를 이용해 내적한 값이 벡터의 차원이 커지면 학습이 잘 안 될 수 있기 떄문에 벡터 크기에 반비례하도록 크기를 조정하는 것이다. 즉, 다음 수식과 같이 어텐션을 적용하게 된다.

![스크린샷 2025-03-30 오후 8 14 31](https://github.com/user-attachments/assets/482ff98e-bc07-4b1b-875f-eca896e30d7c)

- 위 수식처럼 `query`와 `key`를 내적한 값에 `key` 벡터의 차원 수를 제곱근한 값으로 나눈 후 소프트맥스 함수를 적용한다. 여기서 d<sub>k</sub>는 key 벡터의 차원 수를 의미한다. 전체적인 스케일 내적 어텐션을 그림으로 확인해 보자.

![스크린샷 2025-03-30 오후 8 17 22](https://github.com/user-attachments/assets/39480ea0-e069-491c-bcab-6af2504bbc22)

- 그림을 보면 먼저 `query` 벡터와 `key` 벡터에 대해 어텐션 스코어 맵을 만들고 크기를 조정한다. 그리고 선택적으로 마스크(`mask`)를 적용한다(이는 바로 다음에 나오는 함수에서 설명하겠다). 그리고 이 값에 소프트맥스 함수를 적용한 후 마지막으로 value에 대한 가중합을 적용한다. 
- 그림을 보면 `query`, `value`의 내적 부분과 마지막 `value`와의 가중합 부분이 `MatMul`이라는 행렬곱으로 나와 있는데, 이 부분은 코드를 먼저 본 후 어떤 내용인지 알아보자.

```python
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
```

- 먼저 `k(key)`, `q(query)`, `v(value)`에 대한 입력값을 모두 함수의 인자를 통해 받는다. 각 인자값은 모두 문장인데, 각 단어가 벡터로 돼 있고, 이것들이 모여서 행렬로 돼 있는 구조다. 개념적인 어텐션과 실제로 구현된 어텐션의 경우 각 단어의 벡터끼리 연산을 해서 계산하는 구조였다. 하지만 실제 코드와 위의 그림의 경우 행렬로 연산이 이뤄지고 있어서 혼동될 수 있지만 자세히 알아보면 결국 둘 다 똑같은 내용임을 알 수 있다. `query`, `key`, `value` 값이 행렬로 들어오면 어떻게 계산되는지 알아보자.

![스크린샷 2025-03-30 오후 8 24 41](https://github.com/user-attachments/assets/51caa2d1-d58d-4c7a-b09f-e340561ea7bb)

- 위 그림에서 보면 `query` 행렬이 있고 `key`에 대한 행렬이 있다. 각 행렬을 봤을 때 각 단어 벡터에 대한 내적만 연산하고자 한다면 `key` 행렬을 전치 행렬(Transpose matrix)로 만들어 행렬곱 연산을 하면 된다. 행렬곱 연산을 할 때는 `query` 행렬의 행벡터와 전치된 `key` 행렬의 열백터와 내적 연산을 통해 어텐션 점수를 만들어 낼 수 있다. 예를 들어, query `자연어`에서 key `아주`간의 어텐션 점수를 보고자 한다면 `query` 행렬의 열벡터 중 `자연어`에서 `key` 행렬의 열벡터 중 `아주` 간의 어텐션 점수를 보고자 한다면 `query` 행렬의 열백터 중 `자연어` 단어의 벡터와 `key` 행렬의 열백터 중 `아주`라는 단어 벡터의 내적 연산을 해서 만들어 낼 수 있다. 이처럼 `query` 행렬과 `key`에 대한 전치 행렬과 행렬곱 연산을 수행한다면 어텐션 맵으로 만들 수 있다. 이 같은 연산 방법을 선형대수학에서는 외적 연산(outer product)이라 한다.

![스크린샷 2025-03-30 오후 8 31 02](https://github.com/user-attachments/assets/0f98a88a-5f06-4795-90e7-19b6909764a4)

- 위 외적 공식은 행렬 `u`, `v`에 대한 연산을 풀어 놓은 것이다. 앞서 본 예시에서 `query` 행렬을 행렬 `u`로 `key` 행렬을 행렬 `v`로 본다면 `u`에 있는 각 행벡터와 전치된 `v` 행렬의 열벡터와 행렬곱을 통해 각 벡터 간의 내적 연산을 할 수 있음을 알 수 있다.
- 변수 `key`와 `query`를 `tf.matmul`을 이용해 어텐션 맵을 만들고 `outputs` 변수에 할당하게 된다. 이때 변수 `key`는 전치(transpose)된 상태로 연산돼야 하기 때문에 `matmul` 파라미터인 `transpose_b`에 True를 입력한다. `matmul_ok` 변수에 어텐션 점수를 할당하기 전에 점수에 대한 스케일링을 위해 `key` 벡터의 차원 수에 대해 루트를 취한 값을 나눈다. 차원 수는 `key` 행렬의 마지막 차원의 값을 받아오면 된다.
- 이제 각 `query`와 `key` 간의 어텐션 스코어가 있는 어텐션 맵을 구했다. 이 값들을 확률 값으로 만들어야 한다. 따라서 소프트맥스 함수를 적용해 최종 어텐션 맵을 구한다. `query`와 `value` 사이의 관계가 확률값으로 표현됐다. 이제 이 확률값을 각 `value`와 곱해서 가중합을 구하면 된다. 이 역시 행렬로 이뤄져 있기 때문에 간단하게 행렬곱을 통해 구할 수 있다.
- 최종적으로 `query`, `key`, `value`의 관계를 통해 구한 문맥 벡터를 리턴하면 함수가 끝난다. 여기서 한 가지 설명하지 않고 넘어간 것이 있다. 바로 함수의 파라미터인 `mask`인데, 이 내용은 앞서 언급한 마스크 기능에 대한 내용이다. 마스크 기능을 활용하기 위해서는 순방향 마스크 어텐션을 이해할 필요가 있다. 다음 절에서는 순방향 마스크 어텐션을 이해하면서 구현해 보자.

#### 순방향 마스크 어텐션(Subsequent Masked Attention)

- 이전에 구현했던 순환 신경망 기반의 시퀀스 투 시퀀스와의 차이를 다시 생각해보자. 순환 신경망의 경우 스텝이 존재해서 각 스텝마다 단어가 입력으로 들어가는 형태였다. 하지만 그에 반해 트랜스포머 모델의 경우 전체 문장이 한번에 행렬 형태로 입력되는 구조다. 이러한 차이점 때문에 생기는 문제점이 있다. 예를 들어, 문장을 예측하는 과정에서 디코더 부분에 입력이 들어갈 텐데 순환 신역망 구조의 경우 자신보다 앞에 있는 단어만 참고해서 단어를 예측한다. 하지만 트랜스포머의 경우 전체 문장이 들어가기 떄문에 위치와 상관없이 모든 단어를 참고해서 예측할 것이다. 이는 직관적으로도 틀린 상황이다. 자기 자신도 아직 예측을 하지 않았는데 뒤에 있는 단어를 예측한다는 것은 말이 되지 않는다. 따라서 이러한 문제를 해결하기 위해 자신보다 뒤에 있는 단어를 참고하지 않게 하는 기법이 마스크 기법이다.

![스크린샷 2025-03-30 오후 8 41 50](https://github.com/user-attachments/assets/2f75afd9-b1bb-4d24-879b-a11a8cc632fb)

- 예를 들어 '딥러닝 자연어 처리 아주 좋아요'라는 문장이 있고 현재 디코더에는 '딥러닝 자연어 처리' 뒤에 나오는 단어를 예측해야 하는 상황이라 가정하자. 이 상황에서 '처리'라는 단어는 앞서 나온 '딥러닝'과 '자언어' 단어에 관한 관계값은 나타낼 수 있지만 뒤에 나올 '아주', '좋아요'에 대한 정보는 나타나면 안 된다. 따라서 어텐션 맵에 대해 이처럼 '처리'라는 단어 뒤에 나오는 단어에 대한 관계 정보를 보이지 않게 만들 필요가 있다. 
- 이를 위해 어텐션 맵에서 위쪽 삼각형 영역에 마스킹을 씌우는 방법을 취한다. 이 방법을 사용하면 `query` 입장에서 현재 위치 이전에 `key` 정보에 대해서는 관계 정보를 확인할 수 있지만 이후에 나오는 다른 단어 `key` 정보에 대해서는 관계 정보를 보지 못하게 된다. 
- 이제 순방향으로 어텐션 맵을 마스킹하는 방법을 코드를 통해 알아보자.

```python
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
```

- 어텐션 맵을 만드는 방법은 간단하게 앞에서 본 매트릭스 그림과 같이 삼각형 형태의 행렬을 만들고 마스킹할 영역의 값은 1, 아닌 영역은 0으로 만들면 된다. 여기서는 행렬의 아래쪽 삼각형 영역이 0이 되는 하삼각행렬을 만들 텐데, 이를 만드는 방법은 `tf.linalg.bandpart(입력행렬, -1, 0)`을 실행하면 된다. 이 함수에 입력하는 행렬로는 모든 값이 1인 행렬을 만들면 되며, 이 행렬은 `tf.ones` 함수를 통해 만들 수 있다. 이렇게 하삼각행렬을 만들면 마스킹하고자 하는 영역에 1 값을 주기 위해 상수 1로부터 만들어진 하삼각행렬을 뺸다. 상수 1을 뺴는 연산은 행렬에 대해 전체 행렬에 연산이 적용되는 브로드캐스팅 연산이라 보면 된다. 
- 다시 앞에서 본 스케일 내적 어텐션으로 돌아가보자. 앞에서는 마스크가 적용되는 코드 라인을 생략했다. 이제 마스크를 생성할 준비가 끝났으니 잠깐 코드를 보자.

```python
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
```

- 마스크의 목적을 다시 생각해보자. 자신보다 뒤에 나오는 단어를 참조하지 못하도록 행렬의 아래쪽 삼각형 부분을 제외한 값에 마스킹 처리를 해주는 것이었다. 이를 구현하려면 아래쪽 삼각형 부분을 제외한 값에 매우 작은 값을 넣으면 된다. 여기서는 그 값으로 -10의 9승인 '-1e9' 값을 넣는다. 이것은 특별히 의미가 있는 값이 아니라 매우 작은 어떤 값을 넣어주려는 것이다. 이제 앞에서 만든 `mask`를 이용해 `padding`이라는 변수를 만든다. 이 값은 `mask`와 동일한 형태이면서 모든 값이 앞에서 말한 매우 작은 음수값을 가지는 행렬이다. 
- 이렇게 처리하면 어텐션 맵의 아래쪽 삼각형 부분을 제외한 곳은 모두 매우 작은 음수값을 가지고 있을 것이다. 매우 작은 음수값을 넣은 이유는 어텐션 맵을 구한 다음 연산을 생각해보면 된다. 어텐션 맵의 다음 연산은 소프트맥스 함수를 적용하는 것이다. 소프트맥스 함수에 매우 작은 음수값을 넣을 경우 거의 0 값에 수렴하는 값을 갖게 된다. 결국 마스킹 처리된 부분을 참고할 수 없게 되는 것이다. 
- 마스크 어텐션의 경우 내적 함수와 별도로 새로운 함수로 정의하는 것이 아니라 내적 함수 중간에 조건식에 따라 선택적으로 사용할 수 있게 넣어야 한다. 따라서 앞서 설명한 스케일 내적 어텐션 함수를 가지고 마스크 기법을 추가한 내적 함수는 최종적으로 다음과 같이 구성될 것이다. 

```python
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
```

- 이제 이렇게 해서 내적 어텐션 함수를 완전히 구현했다. 마지막으로 멀티 헤드 어텐션의 개념을 알아본 후 코드로 구현한 내용을 알아보자.

#### 멀티 헤드 어텐션

- 셀프 어텐션에 대한 정보를 하나만 생성해서 모델에서 추론할 수도 있다. 하지만 어텐션이 하나가 아닌 여럿에 대한 정보를 줄 수 있다면 어떨까? 멀티 헤드 어텐션은 어텐션 맵을 여럿 만들어 다양한 특징에 대한 어텐션을 볼 수 있게 한 방법이다. 

![스크린샷 2025-03-30 오후 9 06 22](https://github.com/user-attachments/assets/fd976be3-5a6c-441b-99f0-ba6c78092868)

- 앞서 내적 셀프 어텐션에서 본 `query`, `key`, `value`에 대한 특징값을 헤드 수만큼 나눠서 리니어(Linear) 레이러를 거쳐 내적 어텐션(Scaled Dot-Product)을 구해서 다시 합치는 과정을 거친다. 이 과정을 거치고 최종적으로 리니어 레이어를 거쳐 나오면 멀티 헤드 어텐션에 대한 과정을 마치게 된다.
- 멀티 헤드 어텐션의 구현은 다음과 같다.

```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = kargs['num_heads']
        self.d_model = kargs['d_model']

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(kargs['d_model'])
        self.wk = tf.keras.layers.Dense(kargs['d_model'])
        self.wv = tf.keras.layers.Dense(kargs['d_model'])

        self.dense = tf.keras.layers.Dense(kargs['d_model'])

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
```

- 멀티 헤드 어텐션은 기본적으로 앞에서 구현한 내적 어텐션을 기반으로 구현돼 있다. 클래스 구현을 세부적으로 알아보자. 먼저 `__init__` 함수부터 살펴보자.
- `__init__` 함수에서는 입력 파라미터로 `**kargs`를 받으며, 이 안에 `d_model`과 `num_heads`가 있다. 여기서 `d_model`은 어텐션을 연산할 때 `key`, `query`, `value`에 대한 차원을 정의하기 위한 파라미터고 `num_heads`는 어텐션 헤드 수를 정의하기 위한 파라미터다. 각 파라미터는 `self.d_model`, `self.num_heads`에 각각 할당 된다. 
- 항상 `d_model`의 차원 수는 헤드 갯수만큼 나눠져야 하기 때문에 `d_model` 값과 `num_heads`를 나눴을 때 나머지가 발생하면 안 된다. 이러한 문제를 방지하기 위해 `assert` 구문을 활용해 두 변수를 대상으로 나눗셈했을 때 나머지가 발생하면 에러가 발생하도록 구현했다. `d_model`을 `num_heads`로 나눴을 때 나머지가 0이라면 `d_model`의 차원 값을 어텐션 헤드 수만큼 나눈 값을 `self.depth`에 할당해 각 헤드에 입력될 벡터의 차원 수를 정하게 한다. `self.depth`는 뒤에 나올 `split_heads` 함수에서 다룰 예정이다. 
- `self.wq`, `self.wk`, `self.wv`는 앞에서 언급한 스케일 내적 연산 이전에 입력한 `key`, `query`, `value`에 대한 차원 수를 맞추기 위한 레이어다. 이를 위해 `tf.keras.layers.Dense`를 생성해 각 변수에 할당한다. 마지막으로 `self.dense`는 셀프 어텐션 레이어를 출력하기 위한 레이어다. 앞의 레이어와 마찬가지로 `Dense`를 생성하고 할당하게 한다. 멀티헤드 어텐션에 대한 설정을 `__init__` 함수에서 정의했다면 입력한 이번에는 각 벡터를 헤드 수만큼 나눌 수 있게 하는 `split_heads` 함수를 먼저 살펴보자.
- `split_heads`는 `key`, `query`, `value` 에 대한 벡터를 헤드 수만큼 분리할 수 있게 하는 함수다. 이 함수는 \[배치 차원 X 시퀀스 차원 X 피처 차원\]으로 구성된 벡터를 \[배치 차원 X 헤드 차원 X 시퀀스 차원 X 피처 차원\]으로 변환하는 역할을 한다.
- 먼저 각 피처 차원을 헤드 수만큼 분리해야 한다. `tf.reshape`를 활용하면 피처 차원을 헤드 수만큼 분리할 수 있다. `reshape`에 벡터 `x`를 입력하고 변경할 핼렬의 차원값을 (batch_size, -1, self.num_heads, self.depth)로 입력한다. 여기서 배치 크기를 나타내는 batch_size를 왜 함수의 입력 파라미터로 전달하는지 궁금해할 수 있다. `reshape`할 때는 한 개의 차원에 대해서만 값을 알지 않아도 재구성이 가능하다. `num_heads`와 `depth`의 경우는 이미 하이퍼파라미터에 의해 정해져 있지만 배치 크기나 시퀀스 크기에 대해서는 그렇지 못하다. 모델을 학습하는 도중 입력 배치 데이터의 시퀀스 길이가 매번 바뀌는 경우가 있을 수 있기 때문이다. 따라서 정해진 시퀀스가 매번 바뀔 수 있기 때문에 `batch_size` 값을 정의하게 한 것이다. 
- `tf.reshape`만 하게 되면 \[배치 차원 X 시퀀스 차원 X 헤드 차원 X 피처 차원\]으로 구성된다. 원하는 출력 차원으로 바뀌기 위해 `tf.transpose` 연산을 통해 시퀀스, 헤드 차원만 바꿀 수 있게 한다. 

```python
   def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
```

- 이제 본격적으로 멀티헤드 어텐션이 동작하는 `call` 함수를 구현하자. 앞에서 만든 스케일 내적 어텐션 연산과 `__init__`, `split_heads` 함수를 활용해 어텐션 레이어 연산을 구성한다. 입력 파라미터는 입력 벡터 `key`, `query`, `value`, 순방향 마스크를 입력할 `mask`다. 먼저 입력한 key, query, value 값을 self.wq, self.wk, self.wv 레이어를 통해 내적 연산할 준비를 한다. 그리고 self.split_heads 함수로 헤드 수에 맞게 피처를 분리하고 `scaled_dot_product_attention`으로 내적 어텐션을 수행한다. 이렇게 하면 여러 개의 헤드 피처에 대해 어텐션이 적용된 벡터를 얻을 수 있다. 어텐션이 적용된 벡터는 `scaled_attention`을 통해 얻을 수 있다. 그 밖에 `attention_weight`는 시퀀스의 요소별로 서로 어텐션이 얼마나 적용돼 있는지 볼 수 있다. 출력된 `scaled_attention`은 다시 \[배치 차원, 시퀀스 차원, 피처 차원\]으로 맞춰 출력해야 하기 때문에 `tf.transpose`와 `tf.reshape`을 다시 활용한다. 여기서 두 함수를 구현하는 방식은 앞서 `split_heads`에서 구현한 내용을 반대로 만든다고 생각하면 쉬울 것이다. 그리고 마지막으로 `self.dense`를 통해 출력할 멀티 헤드 어텐션 벡터를 구성한다. 

#### 포지션-와이즈 피드 포워드 네트워크

- 트랜스포머 네트워크에서는 셀프 어텐션 레이어를 거친 다음 피드 포워드 네트워크를 거치게 돼 있다. 이 네트워크는 한 문장에 있는 단어 토큰 벡터 각각에 대해 연산하는 네트워크로서 논문에서는 포지션-와이즈 피드 포워드 네트워크(Position-wise Feedforward Network)라 표현한다. 

![스크린샷 2025-03-30 오후 9 31 34](https://github.com/user-attachments/assets/872fab99-fcce-44e6-8d11-7c6762127c6d)

- 위 수식과 같이 네트워크를 구성하는 데 두 개의 리니어 레이어를 거치고 그 사이 활성화 함수로 `Relu` 함수를 활용한다. 이 네트워크의 구현은 다음과 같다.

```python
def point_wise_feed_forward_network(**kargs):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(kargs['dff'], activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(kargs['d_model'])  # (batch_size, seq_len, d_model)
    ])
```

- 여기서는 피드 포워드 네트워크의 연산을 하나로 묶어 주기 위해 `tf.keras.Sequential`을 활용했다. `Sequential` 안의 연산은 리스트에 선언한 순서대로 진행한다고 생각하면 된다. 먼저 입력 시퀀스 벡터를 받아 첫 리니어 레이어를 `Dense`를 통해 거치게 한다. 이 레이어에 대해서만 활성화 함수 `relu`를 정의하고 레이어 차원에 대한 설정을 인자로 지정한다. 첫 레이어를 거치고 다시 `Dense`를 거치면 피드 포워드에 대한 네트워크 연산은 마치게 된다. 이 연산은 `Encoder`, `Decoder`에서 구현되는 것을 다시 확인할 수 있다. 

#### 리지듀얼 커넥션

- 리지듀얼 커넥션(Residual Connection)은 간단하게 다음 그림에 나오는 방법으로 연산을 수행한다. 

![스크린샷 2025-03-30 오후 9 37 14](https://github.com/user-attachments/assets/ad9b0ab7-2ee4-444d-ab4d-f614cd407216)

- 위 그림과 같이 입력 정보 x가 있고 네트워크 레이어를 거쳐 나온 정보 F(x)가 있다고 하면 이 두 정보를 더해 앞에 있던 정보 x를 보존하게 된다. 특별히 트랜스포머 네트워크에서는 리지듀얼 커넥션을 한 다음, 레이어에 대한 값을 노멀라이즈하는 레이어 노멀라이제이션을 수행한다.
- 이 같은 방식으로 다음과 같이 구현할 수 있다. 

```python
self.ffn = point_wise_feed_forward_network(**kargs)
self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
ffn_output = self.dropout2(ffn_output)
out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
```

- 위 코드는 인코데 레이어에서 리지듀얼 커넥션을 수행하는 부분을 보여준다. 먼저 연산할 레이어 피드 포워드 네트워크와 레이어 노멀라이즈를 각각 생성한다. 여기서 레이어 노멀라이즈는 각 레이어별 뉴런을 노멀라이즈하는 역할이라고 이해하면 된다. 피드 포워드 네트워크를 만들기 위해서는 앞서 구현한 `point_wise_feed_forward_netword`를 통해 네트워크를 생성한다. 그리고 레이어 노멀라이즈를 위해 `tf.keras.layers.LayerNormalization`을 생성한다. 이렇게 생성한 레이어들을 `self.ffn`, `self.layernorm2`에 할당한다. 여기서 `self.ffn`이 앞서 설명한 F(x)에 해당한다. 
- 할당한 각 레이어를 가지고 리지듀얼 커넥션 연산을 수행하는 방법을 보자. 입력 벡터 `out1`을 먼저 `self..ffn`을 통해 피드 포워드 연산을 하게 해서 ffn_output으로 할당한다. 그다음 입력 벡터 `out1`과 피드 포워드 네트워크를 거쳐 나온 `ffn_output`을 더해 리지듀얼 커넥션을 수행한다. 이어서 리지듀얼 컨넥션을 한 값을 `self.layernorm2`에 입력해 레이어 노멀라이즈 연산을 수행해 `out2`에 할당한다. 
- 이렇게 레이어 노멀라이제이션 함수까지 정의했다면 하나의 모듈인 리지듀얼 커넥션과 레이어 노멀라이제이션을 사용할 수 있게 된다. 이 함수는 앞에서 모델을 나타낸 그림 6.19 '트린스포머 네트워크의 전체 구조'에서 `Add & Norm`에 해당하는 부분으로 자주 사용된다. 

#### 포지션 인코딩

- 앞의 순환 신경망과 달리 트랜스포머 모델은 입력을 단어 하나하나 순차적으로 넣지 않고 한번에 넣게 된다. 따라서 입력 시퀀스 정보에 대한 순서 정보를 부가적으로 주입할 필요가 있다. 포지션 인코딩(Positional Encoding)은 트랜스포머 모델에 순서 정보갸ㅏ 반영되지 않는 문제를 해결하기 위해 사용한 기법이다. 
- 순서 정보를 주입하는 방법으로 여러 가지 방법이 있지만 논문에서 선택한 방법은 아래의 함수 값을 입력값에 추가로 넣는 방법이다. 

![스크린샷 2025-03-30 오후 9 49 29](https://github.com/user-attachments/assets/7d3c3f62-3843-4222-b0e6-7d3cb581af88)

- 포지션 인코딩에 대한 수식은 크게 2개로 나뉜다. 첫 번째 수식은 피처 차원에서 인덱스가 짝수인 경우에 대해 사인 함수값을 할당하는 것이고, 두 번째 수식은 인덱스가 홀수인 경우에 대해 코사인 함수값을 할당하는 것이다. 사인 함수와 코사인 함수 안에 있는 식은 각 시퀀스 위치에 따라 피처 차원 인덱스에 각자 위치 정보를 달리 주고자 하는 의도를 가지고 있다. 포지션 인코딩을 코드로 구현하면 다음과 같다. 

```python
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * i//2) / np.float32(d_model))
    return pos * angle_rates
```

- 포지션 임베딩을 만들 행렬을 구성하기 위해 `get_angles` 함수를 선언한다. 이 함수는 앞에서 본 수식 중 `pos/100002i/dim` 값을 만드는 함수다. 각 입력 파라미터는 pos, i, d_model로 돼 있다. pos에는 포지션에 대한 인덱스 위치 리스트를 입력하고 i에는 차원에 대한 리스트를 입력한다. 이렇게 하면 각 순서에 따른 각도값을 얻을 수 있다. 이제 `get_angles`를 가지고 포지션 임베딩을 만들어본다. 

```python
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
```

- `get_angles` 함수의 결괏값을 출력하면 포지션과 차원별로 각기 다른 값이 순차적으로 할당돼 있을 것이다. 이제 이 값들을 짝수 차원에는 사인 함수를, 홀수 차원에서는 코사인 함수를 적용해 보자. 짝수 차원과 홀수 차원에 적용하려면 배열의 인덱스에 \[0::2\], \[1::2\]와 같은 방식으로 입력하면 된다. 이렇게 하면 \[0::2\]는 0에서부터 2씩 증가하는 인덱스의 배열 값만 출력되고 \[1::2\]는 1에서부터 2씩 증가하는 인덱스의 배열 값만 출력된다. 이렇게 사인, 코사인 함수를 적용한 값은 배열 `angle_rads`에 할당된다. 마지막으로 나중에 인코더 디코더 레이어에서 바로 활용할 수 있도록 배치 차원을 만들기 위해 다음과 같이 한다. 
- `np.newaxis`는 새로운 차원을 만드는 역할을 한다. 이렇게 `newaxis`를 인덱스 칸에 추가하면 새로운 차원을 만들게 된다. 새로운 차원을 만들면 만들어진 포지션 임베딩 행렬의 차원은 \[배치 차원 X 시퀀스 차원 X 피처 차원\]으로 구성된다.
- 이렇게 구성을 마친 배열은 `tf.cast`를 통해 상수가 되어 모델 그래프에 활용할 수 있게 된다. 이렇게 만들어진 포지션 임베딩은 워드 임베딩과 더해져 각 워드 정보에 순서가 들어갈 수 있도록 활용된다. 
- 이제 포지션 인코딩 기법까지 모든 모듈에 대한 정의가 끝났다. 이렇게 정의한 모듕을 이용해 인코더, 디코더 모듈을 정의해보자.

#### 인코더
- 인코더는 앞서 순환신경망 기반 시퀀스 투 시퀀스와 같이 모델의 입력값에 대한 정보를 추출하는 모듈이다.
  
![스크린샷 2025-03-30 오후 10 01 40](https://github.com/user-attachments/assets/5ef99f6b-bff1-48bd-8221-4795367e0ef9)

- 인코더 모듈은 멀티 헤드 어텐션 레이어와 피드 포워드 네트워크로 한 레이어를 구성하고, 각 레이어에는 리지듀얼 커넥션이 함께 적용돼 있다. 앞서 구현한 모듈을 통해 인코더 모듈을 구현해 보자.

```python
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(**kargs)
        self.ffn = point_wise_feed_forward_network(**kargs)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(kargs['rate'])
        self.dropout2 = tf.keras.layers.Dropout(kargs['rate'])

    def call(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
```

- 인코더 레이어는 크게 4가지 레이어로 구성한다. 
  - 멀티 헤드 어텐션
  - 포지션 와이드 피드 포워드 네트워크
  - 레이어 노멀라이제이션
  - 드롭아웃
- 각 레이어에 대해서는 이미 앞에서 다뤘다. 이제 앞에서 배운 모듈들을 인코더 레이어에서 활용하게 해보자.
- `__init__` 함수부터 살펴보자. 먼저 `kargs` 파라미터를 살펴보면 멀티 헤드 어텐션에 적용될 출력 차원 수 `d_model`과 헤드 수 `num_heads`를 적용한다. 그리고 포지션 와이즈 피드 포워드 네트워크의 차원 수 d<sup>ff</sup>를 적용한다. 이 세 파라미터로 필요한 레이어의 출력 차원 수를 정의한다. 이 함수에서는 앞에서 언급한 4가지에 대한 레이어를 생성하는 역할을 하게 될 것이다. `MultiHeadAttention` 클래스를 활용해 멀티 헤드 어텐션 레이어를 생성하고, `position_wise_feed_forward_network`를 활용해 포지션 와이즈 피드 포워드 네트워크를 생성한다. 그리고 나머지 `tf.keras.layers.LayerNormalization`과 `tf.keras.layers.Dropout`을 이용해 레이어 노멀라이제이션과 드롭아웃 레이어를 생성한다. 이렇게 생성한 레이어들을 `call` 함수에서 구체적으로 연산할 수 있도록 구현해보자.
- `call` 함수에서는 크게 입력 벡터인 `x`와 패딩 마스크 `mask`를 입력 파라미터로 정의한다. 입력한 파라미터는 먼저 멀티 헤드 어텐션 레이어인 `self.mha`를 거친다. 여기서는 셀프 어텐션을 하기 때문에 멀티 헤드 어텐션에 들어가는 모든 key, query, value는 동일하다. 그리고 리지듀얼 커넥션을 하기 위해 멀티 헤드 어텐션에서 나온 출력값 `attn_output`과 입력값 `x`를 더하고 `self.layernorm1`을 통해 노멀라이제이션을 수행한다. 노멀라이제이션까지 연산을 마친 결과값은 `out1`에 할당한다. 
- 멀티 헤드 어텐션을 하고 나면 포지션 와이즈 피드 포워드 네트워크를 연산한다. 앞서 연산한 방식과 똑같이 앞에서 연산을 수행한 변수 `out1`을 입력으로 `self.ffn`을 거쳐 `ffn_output`으로 출력하게 한다. 그런 다음 `out1`과 `ffn_output`을 더해 리지듀얼 커넥션을 하고 다시 레이어 노멀라이제이션을 한다. 이렇게 노멀라이제이션한 값은 `out2`에 할당하고 인코더 레이어에 출력하게 된다.
- 추가로 위의 전체 과정에서 `dropout`은 멀티 헤드 어텐션과 포지션 와이드 피드 포워드 네트워크 연산을 할 떄 한 번씩 적용해 모델에 제너럴라이제이션을 할 수 있게 한다. 
- 이렇게 인코더를 구현하면 여러 개의 인코더 레이어를 쌓을 수 있게 준비된 것이다. 이제 `Encoder`라는 클래스로 인코더 레이어를 쌓고, 워드 임베딩과 포지션 임베딩 정보를 받아 텍스트에 대한 컨텍스트 정보를 만들어 보자.

```python
class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(Encoder, self).__init__()

        self.d_model = kargs['d_model']
        self.num_layers = kargs['num_layers']

        self.embedding = tf.keras.layers.Embedding(kargs['input_vocab_size'], self.d_model)
        self.pos_encoding = positional_encoding(kargs['maximum_position_encoding'], 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(**kargs) 
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(kargs['rate'])

    def call(self, x, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x  # (batch_size, input_seq_len, d_model)
```

- 전체적으로 Encoder 클래스에서는 인코더 레이어, 워드 임베딩, 포지션 인코더 등으로 구성되어 연산을 수행한다. 여기서 인코더 레이어는 여러 개의 레이어로 구성할 수 있다. 
- `__init__` 함수부터 살펴보자. 입력 파라미터부터 살펴보면 인코더와 관련된 파라미터인 `kargs`가 먼저 있다. `kargs`를 살펴보면 인코더 레이어 수인 `num_layers`와 워드 임베딩과 포지션 임베딩 차원 수를 결정하는 `d_model`이 있다. 그리고 워드 임베딩의 사전 수를 입력하는 `input_vocab_size`, 포지션 인코더의 최대 시퀀스 길이를 지정하는 `maximum_position_encoding`이 있다. 
- 이 파라미터로 인코더를 구성하면 된다. 먼저 워드 임베딩부터 살펴보면 `tf.keras.layers.Embedding`으로 워드 임베딩 레이어를 생성해 `self.embedding`에 할당한다. 그리고 `positional_encoding` 함수를 통해 포지션 임베딩을 `self.pos_encoding`에 할당한다. 가장 중요한 인코더 레이어는 레이어 개수만큼 인코더 레이어를 생성해 `self.enc_layer`에 할당한다. 마지막으로 드롭아웃 레이어를 생성한다.
- `__init__` 함수에서 생성한 각 레이어를 가지고 `call` 함수에서 인코더 연산을 구현해 보자. 먼저 입력한 벡터에 대한 시퀀스 길이를 받는다. 이 변수는 포지션 임베딩을 위해 만든 것이다. 포지션 임베딩의 경우 행렬의 크기가 고정돼 있고, 워드 임베딩의 경우 입력 길이에 따라 가변적이다. 따라서 포지션 임베딩의 경우 워드 임베딩과 더할 경우 워드 임베딩의 길이에 맞게 행렬 크기를 조절해야 한다. 이때 시퀀스 길이를 활용하게 된다.
- 그 다음 `self.embedding`을 통해 워드 임베딩을 할당받는다. 임베딩이 할당된 후에 임베딩 차원수의 제곱근만큼에 대한 가중치를 곱한다. 이 연산 과정은 각 워드 임베딩에 대한 스케일을 맞추기 위한 것으로 보면 된다. 가중치 값이 적용된 워드 임베딩은 `self.pos_encoding`을 통해 포지션 임베딩 정보를 더하게 된다.
- 워드 임베딩에 포지션 임베딩을 더했다면 이제 인코더 레이어로 입력한다. 입력하기 전에 `self.dropout`을 통해 먼저 드롭아웃을 적용하고, 인코더 레이어에 순차적으로 연산을 진행한다. 앞서 인코더 레이어를 배열에 생성해 뒀기 때문에 반복문을 통해 레이어 연산을 거치게 한다. 
- 이제 인코더 부분의 구현이 모두 끝났다. 이제 디코더 모듈을 정의해 보자.


#### 디코더

- 디코더의 경우 인코더와는 조금 다르게 하나의 레이어가 더 추가된 형태다. 우선 인코더 부분과 어떻게 다른지 모델을 나타낸 그림을 통해 알아보자.

![스크린샷 2025-03-31 오후 9 28 32](https://github.com/user-attachments/assets/1c701e38-d5ae-4bc9-b80e-6e168210c60b)

- 보다시피 디코더 부분 역시 전체적인 구조는 인코더와 거의 비슷하게 구성돼 있다. 다른 점은 인코더의 경우 하나의 블록이 2개의 레이어로 구성돼 있는데 디코더의 경우 총 3개의 레이어로 구성돼 있다는 점이다. 즉, 두 개의 어텐션과 하나의 피드 포워드 레이어로 구성돼 있다. 먼저 각 어텐션의 역할을 간략하게 설명하면 첫 번째 어텐션의 경우 그림을 보면 입력이 해당 블록이 아닌 외부에서 들어오는 것을 볼 수 있다. 이 값에는 인코더의 결과값이 들어오게 된다. 즉, 인코더와 디코더의 관계를 확인하는 어텐션 구조다. 
- 그리고 추가로 첫 번째 셀프 어텐션의 경우 그림을 보면 마스크 어텐션이라고 돼 있는 것을 확인할 수 있다. 즉, 예측을 해야 하는 디코더에서는 특정 단어 이후의 단어를 참고하지 않도록 앞에서 말한 마스크 기법을 사용하는 것이다. 이렇게 두 개의 어텐션을 적용한 후 이후 마지막으로 인코더 부분과 동일하게 피드 포워드 레이어를 적용하면 하나의 디코더 블록이 완성된다. 위의 구조를 코드로 구현해 보자.

```python
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(**kargs)
        self.mha2 = MultiHeadAttention(**kargs)

        self.ffn = point_wise_feed_forward_network(**kargs)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(kargs['rate'])
        self.dropout2 = tf.keras.layers.Dropout(kargs['rate'])
        self.dropout3 = tf.keras.layers.Dropout(kargs['rate'])
    
    
    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
```

- 앞서 구현한 인코더 레이어와 구조는 대략 비슷하다. 차이점은 디코더 셀프 어텐션 연산에 순방향 마스크가 입력값으로 추가됐고 어텐션 연산 시 인코더 정보 벡터가 입력으로 들어와 디코더의 정보 벡터와 어텐션 연산을 하는 부분이 추가됐다는 것이다.
- `__init__` 함수의 구현부터 보면 앞에서 구현한 인코더 레이어와 같은 모듈을 생성한다. 여기서 `encoder` 정보와 어텐션 연산을 해야 하기 때문에 멀티 헤드 어텐션과 레이어 노멀라이제이션, 드롭아웃이 하나씩 더 추가됐다. 디코더의 `call` 함수는 `__init__` 함수와 같이 인코더 레이어와 비슷하게 구현한다. 다른 점은 인코더 정보 벡터가 추가로 입력되고, 순방향 어텐션 마스크가 입력으로 들어온다는 점이다. `call` 함수의 파라미터를 살펴보면 디코더 입력 벡터 `x`와 인코더 정보 벡터 `enc_output`, 순방향 어텐션 마스크인 `look_ahead_mask`, 패딩 마스크인 `padding_mask`가 있다. 
- 첫 어텐션 연산은 디코더 벡터를 가지고 셀프 어텐션 연산을 하는 부분이다. 인코더 연산과 같이 멀티 헤드 어텐션을 거쳐 리지듀얼 커넥션을 하고 레이어 노멀라이즈를 하는 과정을 거친다. 
- 다음 연산 과정은 인코더 정보와 디코더 정보를 가지고 어텐션을 적용하는 과정이다. 여기서는 셀프 어텐션이 아닌 인코더의 결괏값과의 관계를 확인하는 어텐션 구조다.

![스크린샷 2025-03-31 오후 9 41 23](https://github.com/user-attachments/assets/37399a77-67c3-45b4-aa87-ccfe688574b6)

- 구체적으로 셀프 어텐션 연산과 다른 점을 확인해보면 함수의 `query`의 경우 디코더의 셀프 어텐션과 레이어의 결괏값이 들어가게 되는데 `key`와 `value`의 경우 인코더의 결괏값이 들어간다는 것이다. 이렇게 연산하게 되면 디코더 값에 대해 인코더의 결괏값과의 관계를 확인하는 것이다. 위 그림을 살펴보면 디코더에서 들어온 '딥러닝, '자연어', '처리'라는 `query` 정보와 인코더에서 들어온 '딥러닝, '자연어', '처리', '어때, '?'라는 `key` 정보가 있으면 이 둘 간의 어텐션 정보를 만든다. 그리고 이 정보를 인코더에서 들어온 정보를 `value`로 해서 가중합을 구하게 한다. 이렇게 하면 '자연어' 다음 단어를 예측할 때 활용할 수 있는 인코더 정보가 되는 것이다. 여기서는 순방향 어텐션을 하지 않기 때문에 패딩 마스크만 적용한다. 
- 인코더 정보와 어텐션 연산을 거치게 되면 다시 리지듀얼 커넥션을 하고 인코더와 같은 방식의 피드 포워드 네트워크 연산을 거치게 된다. 
- 이렇게 연산을 구현함으로써 디코더 레이어에 대해서도 구현을 마쳤다. 이제 디코더 모듈을 구현해 보자.

```python
class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(Decoder, self).__init__()

        self.d_model = kargs['d_model']
        self.num_layers = kargs['num_layers']

        self.embedding = tf.keras.layers.Embedding(kargs['target_vocab_size'], self.d_model)
        self.pos_encoding = positional_encoding(kargs['maximum_position_encoding'], self.d_model)

        self.dec_layers = [DecoderLayer(**kargs) 
                           for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(kargs['rate'])

    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
```

- 디코더 구현은 앞서 인코더 구현과 동일하다. 차이가 있다면 인코더 정보 벡터와 순방향 어텐션 마스크를 추가로 입력받는다는 것이다. 이렇게 하면 트랜스포머 네트워크를 위한 인코더 디코더 모듈이 완성된다. 

#### 트랜스포머 모델 클래스 구현

- 이제 트랜스포머 모델에 대한 인코더와 디코더 모듈의 구현이 끝났다. 마지막으로 이 두 모듈을 이어서 시퀀스 투 시퀀스 모델 형태로 구현하면 된다. 전체 코드 구현은 다음과 같다. 

```python
class Transformer(tf.keras.Model):
    def __init__(self, **kargs):
        super(Transformer, self).__init__(name=kargs['model_name'])
        self.end_token_idx = kargs['end_token_idx']
        
        self.encoder = Encoder(**kargs)
        self.decoder = Decoder(**kargs)

        self.final_layer = tf.keras.layers.Dense(kargs['target_vocab_size'])

    def call(self, x):
        inp, tar = x

        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
        enc_output = self.encoder(inp, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, _ = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output
    
    def inference(self, x):
        inp = x
        tar = tf.expand_dims([STD_INDEX], 0)

        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)        
        enc_output = self.encoder(inp, enc_padding_mask)
        
        predict_tokens = list()
        for t in range(0, MAX_SEQUENCE):
            dec_output, _ = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
            final_output = self.final_layer(dec_output)
            outputs = tf.argmax(final_output, -1).numpy()
            pred_token = outputs[0][-1]
            if pred_token == self.end_token_idx:
                break
            predict_tokens.append(pred_token)
            tar = tf.expand_dims([STD_INDEX] + predict_tokens, 0)
            _, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
            
        return predict_tokens
```

- 앞서 `Seq2Seq` 모델에서 구현한 모습과 크게 다르지 않지만 여기서는 트랜스포머 네트워크로 구성돼 있다. 레이어는 크게 `Encoder`, `Decoder`, 단어 인덱스를 출력할 피드 포워드 네트워크로 구성돼 있다. 이 모델에서는 크게 `call` 함수와 `inference` 함수를 선언한다. `call` 함수의 경우 학습할 때를 위해 활용되는 함수이고 `inference` 함수는 입력 텍스트에 대한 모델 추론을 위한 함수다. 먼저 `__init__` 함수부터 차근차근 함수의 구현 내용을 살펴보자.
- 트랜스포머 모델 클래스에서는 앞서 언급한 3개의 레이어를 생성한다. 여기서 피드 포워드 레이어의 경우 출력 차원이 단어 사전의 토큰 수만큼 돼야 한다. 그리고 클래스의 멤버 변수로 `self.end_token_idx`를 선언한다. 이 변수는 마지막 `end_token`에 대한 인덱스 값을 저장하는 변수로서 `inference` 함수에서 `end_token` 이후로는 더 이상 모델 추론을 하지 않게 하는 역할을 한다.
- `call` 함수에서는 인코더에 입력될 값과 디코더에 입력될 값이 함께 입력 파라미터 `x`에 들어간다. 인코더와 디코더에 들어갈 값은 각각 `inp`, `tar` 변수에 할당한다. 이렇게 할당한 값을 가지고 먼저 패딩 마스크와 디코더에서 사용할 순방향 마스크를 `create_mask`를 통해 받는다. 마스크 값이 준비되면 이제 입력값이 인코더, 디코더, 피드 포워드 네트워크를 거쳐 출력될 수 있게 구현한다. 이렇게 하면 출력에서는 입력에 대한 디코더 출력값을 받을 수 있게 된다. 그리고 이 출력 값을 정답 데이터와 비교해서 손실값을 구하게 될 것이다. 
- `inference` 함수는 `call` 함수와 비슷하지만 다른 점이 하나 있다. `call` 함수의 경우 디코더에 입력할 시퀀스가 주어지지만 `inference` 함수에서는 디코더에 입력할 시퀀스를 매번 생성해야 한다. 그렇기 때문에 `inference` 시작 당시에 시작 토큰만 가지고 디코더에서 다음 토큰을 생성할 수 있게 한다. 이 생성 과정을 계속 반복할 수 있도록 for문을 뒀고, 만약 `end_token`이 나온다면 생성을 멈추게 했다. 이렇게 해서 디코더에서 생성된 토큰들을 출력하게 했다. 
- 이제 이렇게 모델을 구현했다면 실제 모델 객체를 생성할 차례다.

```python
char2idx = prepro_configs['char2idx']
end_index = prepro_configs['end_symbol']
model_name = 'transformer'
vocab_size = prepro_configs['vocab_size']
BATCH_SIZE = 2
MAX_SEQUENCE = 25
EPOCHS = 30
VALID_SPLIT = 0.1

kargs = {'model_name': model_name,
         'num_layers': 2,
         'd_model': 512,
         'num_heads': 8,
         'dff': 2048,
         'input_vocab_size': vocab_size,
         'target_vocab_size': vocab_size,
         'maximum_position_encoding': MAX_SEQUENCE,
         'end_token_idx': char2idx[end_index],
         'rate': 0.1
        }
        
model = Transformer(**kargs)
```

- 모델의 각 하이퍼파라미터는 `kargs` 변수에 `dict` 객체로 저장해 모델에 입력하도록 구성했다. 이렇게 해서 간단히 모델을 생성할 수 있다.
- 이제 모델 객체를 생성할 수 있다. 이 모델 객체를 가지고 모델 학습을 진행해 보자. 모델 학습을 위한 손실 함수 구현 및 모델 컴파일 등의 구현은 앞선 `Seq2Seq` 모델과 동일하다.

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

def loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def accuracy(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)
    pred *= mask    
    acc = train_accuracy(real, pred)

    return tf.reduce_mean(acc)
```

- `loss` 함수와 `accuracy` 함수를 구현하고 `model.compile`에 등록한다. 이전 방식과 동일하게 패딩 영역에 대한 정보를 제외하고 `loss`와 `accuracy`를 반영할 수 있게 구현했다. 

```python
# overfitting을 막기 위한 ealrystop 추가
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)
# min_delta: the threshold that triggers the termination (acc should at least improve 0.0001)
# patience: no improvment epochs (patience = 1, 1번 이상 상승이 없으면 종료)

checkpoint_path = DATA_OUT_PATH + model_name + '/weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create path if exists
if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))
    

cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
```

- `model.compile`을 실행하고 나면 `EarlyStopping`과 `Checkpoint` 콜백 모듈을 선언하고 `model.fit`을 통해 모델 학습을 진행하면 된다. 
- 모델 학습을 마치면 성능 그래프를 출력하자.

```python
history = model.fit([index_inputs, index_outputs], index_targets, 
                    batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])
```

```python
plot_graphs(history, 'accuracy')
```

![스크린샷 2025-03-31 오후 10 19 23](https://github.com/user-attachments/assets/5108d48f-15df-4840-a47b-ba3f6c8fa75f)


```python
plot_graphs(history, 'loss')
```

![스크린샷 2025-03-31 오후 10 19 37](https://github.com/user-attachments/assets/db7fd420-8a0b-4509-b03b-f00a31eb42a0)

- 이전 모델 학습 성능 그래프와 비교하면 다른 점이 하나 있다. 검증 `accuracy`의 경우 계속해서 올라가는 수치를 보여 좋아 보이지만 검증 `loss`는 오히려 높아져서 오버피팅으로 보인다. 
- 모델 학습이 끝났다면 이제 모델 추론을 해보자.

```python
text = "남자친구 승진 선물로 뭐가 좋을까?"
test_index_inputs, _ = enc_processing([text], char2idx)
outputs = model.inference(test_index_inputs)

print(' '.join([idx2char[str(o)] for o in outputs]))
```

```
평소에 필요했던 게 좋을 것 같아요
```

- 모델 추론 방법은 간단하다. 텍스트를 `enc_processing` 함수를 통해 인덱스로 변환하고 `model.inference`를 통해 생성된 토큰 인덱스를 받는다. 그리고 `idx2word`를 통해 인덱스를 텍스트 토큰으로 변환한다. 이렇게 하면 입력 문장을 통해 생성된 문장을 확인할 수 있을 것이다. 
- 이렇게 해서 두 가지 타입의 `Seq2Seq` 모델들을 살펴봤다. 이 모델들은 단순히 한글 대화 데이터셋만 학습할 수 있는 모델은 아니다. 어떤 입력 텍스트를 넣었을 때 어떤 텍스트를 기대할 수 있는지 상상할 수 있는 데이터셋이 있다면 시도해보길 바란다.