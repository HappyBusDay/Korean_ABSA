# 2022 국립국어원 인공 지능 언어 능력 평가
<img width="500" alt="2022 국립국어원" src="https://user-images.githubusercontent.com/73925429/200458285-bb6659d2-eebc-48e1-a768-61906aea5d89.png">

#### 국립국어원 데이터셋(모두의 말뭉치) 신청하기 : [2022 인공지능 언어 능력 평가 말뭉치: ABSA](https://corpus.korean.go.kr/main.do#)

~~F1 score 평가 방식 변경으로 대회 기간이 2022년 11월 09일로 연장.~~

### 최종 순위 예

<img width="900" alt="순위 예시" src="https://user-images.githubusercontent.com/73925429/200461303-85d6bcf5-3d91-4145-a4f1-fd81173c81cd.png">

---

# 코드 정리

<table>
    <thead>
        <tr>
            <th>목록</th>
            <th>파일명</th>
            <th>설명</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>Data 만들기</td>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Data_Augmentation.ipynb">Data_Augmentation.ipynb</a>
            </td>
            <td> Random_Insertion, Random_Swap, Random_Deletion Code</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Back_Translation.ipynb">Back_Translation.ipynb</a>                
            <td> Back_Translation Code</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/MakeData_with_API.ipynb">MakeData_with_API.ipynb</a>     
            <td> Naver Open API를 이용하여 데이터의 라벨을 달아주는 Code </td>
        <tr>
            <td>Model Training</td>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/train.ipynb">train.ipynb</a>     
            <td> Model과 Data를 불러와 학습시키는 Code </td>
        </tr>
        <tr>
            <td>Prediction</td>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/test.ipynb">test.ipynb</a>     
            <td> 학습 시켰던 Model의 Weights를 불러와서 새로운 데이터의 결과값을 예측하는 Code </td>
        </tr>        
        <tr>
            <td rowspan=2>Model Ensemble</td>       
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Manual_Ensemble.ipynb">Manual_Ensemble.ipynb</a>
            <td> Prediction의 결과(jsonl파일)를 불러와서 Hard Voting하는 Code</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Auto_Ensemble.ipynb">Auto_Ensemble.ipynb</a>
            <td>다양한 조합의 결과를 도출하는 Ensemble code</td>
        </tr>
    </tbody>
</table>

---

# 가. 개발 환경

    1. Google Colab Pro
    
    2. AWS(Amazom Web Services) 
       구체적인 개발환경은 requirements.txt 참고

---

# 나. 데이터 예시 
 출처 : 국립국어원, 2022 인공지능 언어 능력 평가 말뭉치: ABSA 

    {"id": "nikluge-sa-2022-train-00001", "sentence_form": "둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.", "annotation": [["본품#품질", ["기어", 16, 18], "negative"]]}
    {"id": "nikluge-sa-2022-train-00002", "sentence_form": "이거 뭐 삐꾸를 준 거 아냐 불안하고, 거금 투자한 게 왜 이래.. 싶어서 정이 확 떨어졌는데 산 곳 가져가서 확인하니 기어 텐션 문제라고 고장 아니래.", "annotation": [["본품#품질", ["기어 텐션", 67, 72], "negative"]]}
    {"id": "nikluge-sa-2022-train-00003", "sentence_form": "간사하게도 그 이후에는 라이딩이 아주 즐거워져서 만족스럽게 탔다.", "annotation": [["제품 전체#일반", [null, 0, 0], "positive"]]}

---

# 다. 데이터 증강 방식

#### 1. Augmentation - RI(Random Insertion): 감탄사와 의성어를 문장 내에 추가하는 방식

       
    ex) 나는 자전거 타는 것을 좋아한다. -> 와! 나는 자전거 타는 것을 좋아한다.
   
        
#### 2. Back-Translation


    원본) 나는 자전거 타는 것을 좋아한다. 

    한국어 -> 프랑스어) J'aime faire du vélo. 

    프랑스어 -> 한국어) 저는 자전거 타는 것을 좋아해요.

   위의 예와 같이 특정 문장을 다른 언어로 번역한 후 다시 한국어로 번역하여 의미는 같지만 형태가 다른 문장을 생성하는 방식
    
#### 3. 외부 API 활용
 
   크롤링한 데이터(출처: 네이버쇼핑, 올리브영)에 대해 NAVER CLOVA Sentiment API를 이용하여 Label을 'neutral'과 'negative'를 부여하는 방식
    
---

# 라. 주요 소스 코드

- ## Model Load: Hugging Face에서 Pre-Trained Model 불러오기 ( pip install transformers )
   
   
    <table>
    <thead>
        <tr>
            <th>목록</th>
            <th>Model</th>
            <th>Pre-Trained Data</th>
            <th>링크(HuggingFace)</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>속성범주 (Category)</td>
            <td> ELECTRA</td>
            <td> 한국어로 된 블로그, 댓글, 리뷰 Data </td>
            <td>
                <a href="https://huggingface.co/kykim/electra-kor-base">kykim/electra-kor-base</a>
        </tr>
        <tr>
            <td> RoBERTa</td>
            <td> Wikipidia, BookCorpus, CommonCrawl data 등 100 languagues로 된 Data </td>
            <td>
                <a href="https://huggingface.co/xlm-roberta-base">xlm-roberta-base</a>
        </tr>
        <tr>
            <td>DeBERTa</td>
            <td>한국어로 된 모두의 말뭉치, 국민청원 등의 Data </td>
            <td>
                <a href="https://huggingface.co/lighthouse/mdeberta-v3-base-kor-further">mdeberta-v3-base-kor-further</a>
        </tr>
        <tr>
            <td>감성범주 (Polarity)</td>
            <td> ELECTRA</td>
            <td> 한국어로 된 블로그, 댓글, 리뷰 Data </td>
            <td>
                <a href="https://huggingface.co/kykim/electra-kor-base">kykim/electra-kor-base</a>
        </tr>
    </tbody>
    </table>
    
   > 속성 범주(Category)와 감성 범주(Polarity의 class 불균형을 해소하기 위해서 각 범주를 분리하여 전처리 및 학습을 진행하였다.     
    
   ```c
    # HuggingFace에서 불러오기
    from transformers import AutoTokenizer, AutoModel
    base_model = "HuggingFace주소"

    Model = AutoModel.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
   ```


- ## Data Load: jsonlload
    데이터가 line별로 저장된 json 파일( jsonl )이기 때문에 데이터 로드를 할 때 해당 코드로 구현함

    ```c
    import json
    import pandas as pd
    def jsonlload(fname, encoding="utf-8"):
        json_list = []
        with open(fname, encoding=encoding) as f:
            for line in f.readlines():
                json_list.append(json.loads(line))
        return json_list
    df = pd.DataFrame(jsonlload('/content/sample.jsonl'))
    ```


- ## Inference: predict_from_korean_form
   predict_from_korean_form 6가지의 방법 중 일부
   
   > [HappyBusDay/Korean_ABSA/code/test.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/test.ipynb)

    - #### 방법 1: Force ( Force evaluation of a Argment )
    
         빈칸 " [ ] " 에 대해서 가장 높은 확률의 카테고리를 강제로 뽑아내는 방법 

         [Force에 대한 설명](https://rdrr.io/r/base/force.html)

         ```c

        def predict_from_korean_form_kelec_forcing(tokenizer_kelec, ce_model, pc_model, data):

            ...

            자세한 코드는 code/test.ipynb 참조

            return data
         ```

 
    - #### 방법 2: DeBERTa(RoBERTa)와 ELECTRA 
        
         모델 별 tokenizer를 이용한 inference 진행하는 방법

         ```c

        def predict_from_korean_form_deberta(tokenizer_deberta, tokenizer_kelec, ce_model, pc_model, data):

            ...

           자세한 코드는 code/test.ipynb 참조

            return data
         ```

     
    - #### 방법 3: Threshold
     
         확률 기반으로 annotation을 확실한 것만 가져오는 방법
         
         확실한 것만 잡고 확률값이 낮은 것은 그냥 " [ ] "으로 결과값 도출 

         ```c

        def predict_from_korean_form_kelec_threshold(tokenizer_kelec, ce_model, pc_model, data):

            ...

           자세한 코드는 code/test.ipynb 참조

            return data
         ```



- ## Pipeline 및 Ensemble

   > [HappyBusDay/Korean_ABSA/code/test.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/test.ipynb)
   
    - #### Pipeline: 여러 모델을 불러 결과값 도출
        
        해당 코드는 **12종류[category{6종류} + polarity{6종류}]의 모델**을 불러옴

        " [ ] " 을 최소화 하기 위해 DeBERTa와 ELECTRA 등 여러 모델의 Weight파일을 불러 진행

        ```c
        def Win():

            print("Deberta!!")

            tokenizer_kelec = AutoTokenizer.from_pretrained(base_model_elec)
            tokenizer_deberta = AutoTokenizer.from_pretrained(base_model_deberta)
            tokenizer_roberta = AutoTokenizer.from_pretrained(base_model_roberta)

            num_added_toks_kelec = tokenizer_kelec.add_special_tokens(special_tokens_dict)
            num_added_toks_deberta = tokenizer_deberta.add_special_tokens(special_tokens_dict)
            num_added_toks_roberta = tokenizer_roberta.add_special_tokens(special_tokens_dict)

            ...    

            자세한 코드는 code/test.ipynb 참조

            return pd.DataFrame(jsonlload('/content/drive/MyDrive/Inference_samples.jsonl'))
        ```
    
    
    - #### Ensemble: 위의 Inference의 결과로 만들어진 jsonl파일을 불러와 Hard Voting을 진행
        > [Ensemble.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Ensemble.ipynb)

        > [Auto_Ensemble.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Auto_Ensemble.ipynb)

        <img width="504" alt="hard voting" src="https://user-images.githubusercontent.com/73925429/200563368-a9845d1c-7065-4913-817f-1be4dbbd30d7.png">

        ( Hard Voting )

        > [그림 출처: Ensemble Learning : Voting and Bagging](https://velog.io/@jiselectric/Ensemble-Learning-Voting-and-Bagging-at6219ae)


---

# 마. Reference

[1] [EDA: Easy Data Augmentation](https://arxiv.org/pdf/1901.11196.pdf): Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." arXiv preprint arXiv:1901.11196 (2019).

[2] [Back-Trainslation](https://proceedings.neurips.cc/paper/2020/file/44feb0096faa8326192570788b38c1d1-Paper.pdf): Xie, Qizhe, et al. "Unsupervised data augmentation for consistency training." Advances in Neural Information Processing Systems 33 (2020): 6256-6268.

[3] [ELECTRA](https://arxiv.org/pdf/2003.10555.pdf): Clark, Kevin, et al. "Electra: Pre-training text encoders as discriminators rather than generators." arXiv preprint arXiv:2003.10555 (2020).

[4] [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf): Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).

[5] [DeBERTa](https://arxiv.org/pdf/2006.03654.pdf): He, Pengcheng, et al. "Deberta: Decoding-enhanced bert with disentangled attention." arXiv preprint arXiv:2006.03654 (2020).

[6] [teddysum/korean_ABSA_baseline](https://github.com/teddysum/korean_ABSA_baseline): GitHub



