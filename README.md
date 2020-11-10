# YTS_project

## Abstract

YTS는 Youtube-to-Summary의 약자이며, 사용자들이 Youtube 영상의 링크를 입력하면 영상 내용을 요약하여 제공하는 것을 목적으로 합니다.

본 프로젝트에서 구현한 YTS 모델의 프로세스는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/62092003/98652242-ad94e580-237e-11eb-92aa-57b49c023290.png)

1. Youtube 링크를 기준으로 음성을 추출하고 변환합니다.
2. 추출한 음성들을 STT(Speech-to-Text) 모델을 활용하여 텍스트로 변환합니다.
3. 변환된 텍스트를 문서 요약 모델(TextRank, BertSum) 등을 활용하여 요약합니다.
4. 요약 결과를 3줄로 출력합니다

    * 2020.11.10 기준, Youtube 링크를 입력할 경우 요약 결과(3줄), 키워드, 워드 클라우드, 전체 텍스트를 출력합니다.
    
    ![image](https://user-images.githubusercontent.com/62092003/98654131-462c6500-2381-11eb-9547-b55af8a0ddb0.png)
    ![image](https://user-images.githubusercontent.com/62092003/98654185-56444480-2381-11eb-9734-fcebaa33dee6.png)

## Model Structure

### STT Model

- 본 프로젝트에서는 STT 모델로 LAS(Listen, Attend and Spell) 모델을 활용하고 있습니다.
- 사전 학습에는 'AIHub 자유발화 음성 데이터'를 활용하였습니다.
- 목적에 맞게 파인 튜닝을 거칠 경우 모델의 성능을 더욱 높일 수 있습니다.

![image](https://user-images.githubusercontent.com/62092003/98652311-cbfae100-237e-11eb-8285-90ebffc9e67d.png)

출처: William Chan(2016)

### Documents Summary Model

- 본 프로젝트에서는 Documents Summary 모델로 추출 요약(Extractive Summary) 방식의 TextRank, BertSum 모델을 활용하고 있습니다.

![image](https://user-images.githubusercontent.com/62092003/98652371-e339ce80-237e-11eb-90ed-c68cc75aed7f.png)

- TextRank 모델의 경우 krwordrank, gensim summary 패키지를 활용합니다.
- BertSum 모델은 사전 학습된 Bert 모델을 기반으로 한 추출 요약 모델입니다. 본문과 요약문 데이터를 활용하여 학습할 수 있습니다.

## Guideline

### Install

```powershell
$ git clone https://github.com/Hugekyung/STT_project.git
```

*  Github에는 용량 관계로 학습된 STT 모델과 BertSum 모델을 포함하고 있지 않습니다.

### Python

본 모델은 Python 3.7 버전을 기반으로 구현하였습니다.

Python 3.x 버전 이상 환경에서 활용하시기를 권장합니다.

### Requires

본 모델을 활용하기 위해서는 다음과 같은 패키지들이 필요합니다.

```
librosa==0.7.0
scipy==1.3.1
numpy==1.17.2
tqdm==4.36.1
torch=1.2.0
python-Levenshtein==0.12.0
youtube_dl
pydub
ffmpeg
krwordrank
gensim
```

### Directory Information

```
BertSum: BertSum 모델 학습 및 실행 코드(Github에는 모델 미포함)
STT: STT 모델 학습 및 실행 코드(Github에는 모델 미포함)
TextRank: TextRank 실행 코드
news_data: 뉴스 데이터 수집 코드
visualization: 시각화(wordcloud), BertSum, Textrank 출력 코드
main.py: 전체 프로세스 실행 코드
```

## Usage

1. '[main.py](http://main.py)' 파일을 실행합니다(wordcloud 결과를 확인하기 위해서는 jupyter notebook 환경에서 실행하기를 권장합니다).
2. 원하는 영상의 Youtube 링크를 입력합니다(아래 시연 영상 참조).

   [![YTS_project 시연 영상](http://img.youtube.com/vi/hVwvxnIUysM/0.jpg)](https://youtu.be/hVwvxnIUysM?t=0s)


## Reference

### 1. STT Model/Code

- 한국어 STT 모델(LAS) 베이스라인 모델([https://github.com/clovaai/ClovaCall](https://github.com/clovaai/ClovaCall))
- AIHub 데이터 전처리([https://github.com/sooftware/KsponSpeech-preprocess](https://github.com/sooftware/KsponSpeech-preprocess))

### 2. Documents Summary Model/Code

- BertSum 모델([https://github.com/nlpyang/BertSum](https://github.com/nlpyang/BertSum))
