# 2022 국립국어원 인공 지능 언어 능력 평가
<img width="500" alt="2022 국립국어원" src="https://user-images.githubusercontent.com/73925429/200458285-bb6659d2-eebc-48e1-a768-61906aea5d89.png">

#### 제공 받은 baseline code : [github-teddysum](https://github.com/teddysum/korean_ABSA_baseline)

#### 국립국어원 데이터셋(모두의 말뭉치) 신청하기 : [2022 인공지능 언어 능력 평가 말뭉치: ABSA](https://corpus.korean.go.kr/main.do#)

### 최종 순위

<img width="900" alt="순위 예시" src="https://user-images.githubusercontent.com/73925429/200461303-85d6bcf5-3d91-4145-a4f1-fd81173c81cd.png">

---

# 개발 환경

    1. Google Colab Pro
    
    2. AWS(Amazom Web Services) , Python
    AWS의 구체적인 개발환경은 requirements.txt 참고

---

# 데이터 예시 
( 출처 : 국립국어원, 2022 인공지능 언어 능력 평가 말뭉치: ABSA )

    {"id": "nikluge-sa-2022-train-00001", "sentence_form": "둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.", "annotation": [["본품#품질", ["기어", 16, 18], "negative"]]}
    {"id": "nikluge-sa-2022-train-00002", "sentence_form": "이거 뭐 삐꾸를 준 거 아냐 불안하고, 거금 투자한 게 왜 이래.. 싶어서 정이 확 떨어졌는데 산 곳 가져가서 확인하니 기어 텐션 문제라고 고장 아니래.", "annotation": [["본품#품질", ["기어 텐션", 67, 72], "negative"]]}
    {"id": "nikluge-sa-2022-train-00003", "sentence_form": "간사하게도 그 이후에는 라이딩이 아주 즐거워져서 만족스럽게 탔다.", "annotation": [["제품 전체#일반", [null, 0, 0], "positive"]]}

---

# 주요 소스 코드

- 코드 1: Hugging Face에서 Pre-Trained Model 불러오기 ( pip install transformers )
```c
from transformers import AutoTokenizer, AutoModel
base_model = "HuggingFace주소"

Model = AutoModel.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
```
    - ELECTRA : [kykim/electra-kor-base](https://huggingface.co/kykim/electra-kor-base)

    - RoBERTa : [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)

    - DeBERTa : [mdeberta-v3-base-kor-further](lighthouse/mdeberta-v3-base-kor-further)


- 코드 2: jsonlload
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
- 코드 3: predoct_from_korean_form

    - 코드 3-1: Forcing ( 빈칸 " [ ] " 에 대해서 가장 높은 확률의 카테고리를 강제로 뽑아내는 방법, 용어는 임의로 설정하였음 )
    
        ```c

        def predict_from_korean_form_kelec_forcing(tokenizer_kelec, ce_model, pc_model, data):
        
            ...

            자세한 코드는 code/test.ipynb 참조

            return data
        ```
            
 
    - 코드 3-2: DeBERTa(RoBERTa)와 ELECTRA Pipeline
    
        ```c
        
            def predict_from_korean_form_deberta(tokenizer_deberta, tokenizer_kelec, ce_model, pc_model, data):
            
                ...

               자세한 코드는 code/test.ipynb 참조
               
                return data
        ```


- 코드 4: Inference 여러 모델 ( " [ ] " 을 최소화 하기 위해 DeBERTa와 ELECTRA 등 여러 모델의 Weight파일을 불러 진행)

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
    
