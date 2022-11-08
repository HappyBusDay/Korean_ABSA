# 2022 국립국어원 인공 지능 언어 능력 평가
<img width="500" alt="2022 국립국어원" src="https://user-images.githubusercontent.com/73925429/200458285-bb6659d2-eebc-48e1-a768-61906aea5d89.png">

#### 제공 받은 baseline code : [github-teddysum](https://github.com/teddysum/korean_ABSA_baseline)

#### 국립국어원 데이터셋(모두의 말뭉치) 신청하기 : [2022 인공지능 언어 능력 평가 말뭉치: ABSA](https://corpus.korean.go.kr/main.do#)

### 최종 순위

<img width="900" alt="순위 예시" src="https://user-images.githubusercontent.com/73925429/200461303-85d6bcf5-3d91-4145-a4f1-fd81173c81cd.png">

---

# 개발 환경

    1. Google Colab Pro
    
    2. AWS(Amazom Web Services) 
    AWS의 구체적인 개발환경은 requirements.txt 참고

---

# 주요 소스 코드

- 코드 1: Hugging Face에서 Pre-Trained Model 불러오기 ( pip install transformers )
```c
from transformers import AutoTokenizer, AutoModel
base_model = "HuggingFace주소"

Model = AutoModel.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
```
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

            ce_model.to(device)
            ce_model.eval()
            for idx, sentence in enumerate(data):
                if idx % 10 == 0:
                    print(idx, "/", len(data))
                form = sentence['sentence_form']
                sentence['annotation'] = []
                if type(form) != str:
                    print("form type is arong: ", form)
                    continue

                tmp = []
                flag = False

                for pair in entity_property_pair:      
                    tokenized_data = tokenizer_kelec(form, pair, padding='max_length', max_length=256, truncation=True)
                    input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
                    attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)

                    with torch.no_grad():
                        _, ce_logits = ce_model(input_ids, attention_mask)                
                        tmp.append( ce_logits[0][0] )
                    ce_predictions = torch.argmax(ce_logits, dim = -1)            
                    ce_result = tf_id_to_name[ce_predictions[0]]

                    if ce_result == 'True':
                        flag = True
                        with torch.no_grad():
                            _, pc_logits = pc_model(input_ids, attention_mask)
                        pc_predictions = torch.argmax(pc_logits, dim=-1)
                        pc_result = polarity_id_to_name[pc_predictions[0]]
                        sentence['annotation'].append([pair, pc_result])

                if flag == False:
                    tmp = torch.tensor(tmp)
                    pair = entity_property_pair[torch.argmax(tmp)]

                    with torch.no_grad():
                        _, pc_logits = pc_model(input_ids, attention_mask)
                    pc_predictions = torch.argmax(pc_logits, dim=-1)
                    pc_result = polarity_id_to_name[pc_predictions[0]]
                    sentence['annotation'].append([pair, pc_result])                

            return data
        ```    
        ```c

        def predict_from_korean_form_kelec_forcing(tokenizer_kelec, ce_model, pc_model, data):

            자세한 코드는 all_code.ipynb 참조

            return data
        ```
            
 
    - 코드 3-2: DeBERTa(RoBERTa)와 ELECTRA Pipeline
    
        ```c
        
            def predict_from_korean_form_deberta(tokenizer_deberta, tokenizer_kelec, ce_model, pc_model, data):

                ce_model.to(device)
                ce_model.eval()
                for idx, sentence in enumerate(data):
                    if idx % 10 == 0:
                        print(idx, "/", len(data))
                    form = sentence['sentence_form']
                    sentence['annotation'] = []
                    if type(form) != str:
                        print("form type is arong: ", form)
                        continue
                        
                    for pair in entity_property_pair:
                        tokenized_data_deberta = tokenizer_deberta(form, pair, padding='max_length', max_length=256, truncation=True)
                        tokenized_data_kelec = tokenizer_kelec(form, pair, padding='max_length', max_length=256, truncation=True)
                        input_ids_deberta = torch.tensor([tokenized_data_deberta['input_ids']]).to(device)
                        attention_mask_deberta = torch.tensor([tokenized_data_deberta['attention_mask']]).to(device)
                        input_ids_kelec = torch.tensor([tokenized_data_kelec['input_ids']]).to(device)
                        attention_mask_kelec = torch.tensor([tokenized_data_kelec['attention_mask']]).to(device)

                        with torch.no_grad():
                            _, ce_logits = ce_model(input_ids_deberta, attention_mask_deberta)
                        ce_predictions = torch.argmax(ce_logits, dim = -1)
                        ce_result = tf_id_to_name[ce_predictions[0]]

                        if ce_result == 'True':
                            with torch.no_grad():
                                _, pc_logits = pc_model(input_ids_kelec, attention_mask_kelec)
                            pc_predictions = torch.argmax(pc_logits, dim=-1)
                            pc_result = polarity_id_to_name[pc_predictions[0]]
                            sentence['annotation'].append([pair, pc_result])

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

    test_data = jsonlload(test_data_path)

    entity_property_test_data_deberta, polarity_test_data_deberta = get_dataset(test_data, tokenizer_deberta, max_len_debe)
    entity_property_test_data_kelec, polarity_test_data_kelec = get_dataset(test_data, tokenizer_kelec, max_len_elec)

    entity_property_test_dataloader = DataLoader(entity_property_test_data_deberta, shuffle=True,
                                batch_size=batch_size)

    polarity_test_dataloader = DataLoader(polarity_test_data_kelec, shuffle=True,
                                                  batch_size=batch_size)
    
    model = DebertaBaseClassifier(len(tf_id_to_name), len(tokenizer_deberta))
    model.load_state_dict(torch.load(test_category_extraction_model_path_deberta, map_location=device))
    model.to(device)
    model.eval()
            
    polarity_model = ElectraBaseClassifier_Pola_Base(len(polarity_id_to_name), len(tokenizer_kelec))
    polarity_model.load_state_dict(torch.load(test_polarity_classification_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form_deberta(tokenizer_deberta ,tokenizer_kelec, model, polarity_model, copy.deepcopy(test_data))
    df_pred_first = pd.DataFrame(pred_data)

    with open('Blank1.jsonl', 'w') as file:
        for i in range( len(df_pred_first) ):
            if len(df_pred_first['annotation'][i]) == 0 :
                tmp = '[["제품 전체#일반", [null, 0, 0], "positive"]]'
                file.write(  '{'+'\"id\": \"{0}\", \"sentence_form\": \"{1}\", \"annotation\": {2}'\
                    .format( df_pred_first['id'][i]  ,   df_pred_first['sentence_form'][i], tmp ) +'}' ) 
                file.write("\n")

    print("K up!!")
    
    test_data_blank_first = jsonlload(test_data_path_blank_first)

    entity_property_test_data_kelec, polarity_test_data_kelec = get_dataset(test_data_blank_first, tokenizer_kelec, max_len_elec)

    entity_property_test_dataloader = DataLoader(entity_property_test_data_kelec, shuffle=True,
                                batch_size=batch_size)

    polarity_test_dataloader = DataLoader(polarity_test_data_kelec, shuffle=True,
                                                  batch_size=batch_size)
    
    model = ElectraBaseClassifier_Cate_hiddenup(len(tf_id_to_name), len(tokenizer_kelec))
    model.load_state_dict(torch.load(test_category_extraction_model_path_k_up, map_location=device))
    model.to(device)
    model.eval()
            
    polarity_model = ElectraBaseClassifier_Pola_Base(len(polarity_id_to_name), len(tokenizer_kelec))
    polarity_model.load_state_dict(torch.load(test_polarity_classification_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form_kelec(tokenizer_kelec, model, polarity_model, copy.deepcopy(test_data_blank_first))
    df_pred_second = pd.DataFrame(pred_data)

    with open('Blank2.jsonl', 'w') as file:
        for i in range( len(df_pred_second) ):
            if len(df_pred_second['annotation'][i]) == 0 :
                tmp = '[["제품 전체#일반", [null, 0, 0], "positive"]]'
                file.write(  '{'+'\"id\": \"{0}\", \"sentence_form\": \"{1}\", \"annotation\": {2}'\
                    .format( df_pred_second['id'][i]  ,   df_pred_second['sentence_form'][i], tmp ) +'}' ) 
                file.write("\n")

    print("Original!!")

    test_data_blank_second = jsonlload(test_data_path_blank_second)

    entity_property_test_data_kelec, polarity_test_data_kelec = get_dataset(test_data_blank_second, tokenizer_kelec, max_len_elec)

    entity_property_test_dataloader = DataLoader(entity_property_test_data_kelec, shuffle=True,
                                batch_size=batch_size)

    polarity_test_dataloader = DataLoader(polarity_test_data_kelec, shuffle=True,
                                                  batch_size=batch_size)
    
    model = ElectraBaseClassifier_Cate_Base(len(tf_id_to_name), len(tokenizer_kelec))
    model.load_state_dict(torch.load(test_category_extraction_model_path_origin, map_location=device))
    model.to(device)
    model.eval()
            
    polarity_model = ElectraBaseClassifier_Pola_Base(len(polarity_id_to_name), len(tokenizer_kelec))
    polarity_model.load_state_dict(torch.load(test_polarity_classification_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form_kelec(tokenizer_kelec, model, polarity_model, copy.deepcopy(test_data_blank_second))
    df_pred_third = pd.DataFrame(pred_data)

    with open('Blank3.jsonl', 'w') as file:
        for i in range( len(df_pred_third) ):
            if len(df_pred_third['annotation'][i]) == 0 :
                tmp = '[["제품 전체#일반", [null, 0, 0], "positive"]]'
                file.write(  '{'+'\"id\": \"{0}\", \"sentence_form\": \"{1}\", \"annotation\": {2}'\
                    .format( df_pred_third['id'][i]  ,   df_pred_third['sentence_form'][i], tmp ) +'}' ) 
                file.write("\n")
    
    print("K Dr!!!")

    test_data_blank_third = jsonlload(test_data_path_blank_third)

    entity_property_test_data_kelec, polarity_test_data_kelec = get_dataset(test_data_blank_third, tokenizer_kelec, max_len_elec)

    entity_property_test_dataloader = DataLoader(entity_property_test_data_kelec, shuffle=True,
                                batch_size=batch_size)

    polarity_test_dataloader = DataLoader(polarity_test_data_kelec, shuffle=True,
                                                  batch_size=batch_size)
    
    model = ElectraBaseClassifier_Cate_dr05(len(tf_id_to_name), len(tokenizer_kelec))
    model.load_state_dict(torch.load(test_category_extraction_model_path_k, map_location=device))
    model.to(device)
    model.eval()
            
    polarity_model = ElectraBaseClassifier_Pola_Base(len(polarity_id_to_name), len(tokenizer_kelec))
    polarity_model.load_state_dict(torch.load(test_polarity_classification_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form_kelec(tokenizer_kelec, model, polarity_model, copy.deepcopy(test_data_blank_third))
    df_pred_fourth = pd.DataFrame(pred_data)

    with open('Blank4.jsonl', 'w') as file:
        for i in range( len(df_pred_fourth) ):
            if len(df_pred_fourth['annotation'][i]) == 0 :
                tmp = '[["제품 전체#일반", [null, 0, 0], "positive"]]'
                file.write(  '{'+'\"id\": \"{0}\", \"sentence_form\": \"{1}\", \"annotation\": {2}'\
                    .format( df_pred_fourth['id'][i]  ,   df_pred_fourth['sentence_form'][i], tmp ) +'}' ) 
                file.write("\n")

    print("Gpu LR!!")

    test_data_blank_fourth = jsonlload(test_data_path_blank_fourth)

    entity_property_test_data_kelec, polarity_test_data_kelec = get_dataset(test_data_blank_fourth, tokenizer_kelec, max_len_elec)

    entity_property_test_dataloader = DataLoader(entity_property_test_data_kelec, shuffle=True,
                                batch_size=batch_size)

    polarity_test_dataloader = DataLoader(polarity_test_data_kelec, shuffle=True,
                                                  batch_size=batch_size)
    
    model = ElectraBaseClassifier_Cate_Base(len(tf_id_to_name), len(tokenizer_kelec))
    model.load_state_dict(torch.load(test_category_extraction_model_path_gpu_lr, map_location=device))
    model.to(device)
    model.eval()
            
    polarity_model = ElectraBaseClassifier_Pola_Base(len(polarity_id_to_name), len(tokenizer_kelec))
    polarity_model.load_state_dict(torch.load(test_polarity_classification_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form_kelec(tokenizer_kelec, model, polarity_model, copy.deepcopy(test_data_blank_fourth))
    df_pred_fifth = pd.DataFrame(pred_data)

    with open('Blank5.jsonl', 'w') as file:
        for i in range( len(df_pred_fifth) ):
            if len(df_pred_fifth['annotation'][i]) == 0 :
                tmp = '[["제품 전체#일반", [null, 0, 0], "positive"]]'
                file.write(  '{'+'\"id\": \"{0}\", \"sentence_form\": \"{1}\", \"annotation\": {2}'\
                    .format( df_pred_fifth['id'][i] ,   df_pred_fifth['sentence_form'][i], tmp ) +'}' ) 
                file.write("\n")

    print("The Last Forcing!!")

    test_data_final = jsonlload(test_data_path_blank_fifth)

    entity_property_test_data_kelec, polarity_test_data_kelec = get_dataset(test_data_final, tokenizer_kelec, max_len_elec)

    entity_property_test_dataloader = DataLoader(entity_property_test_data_kelec, shuffle=True,
                                batch_size=batch_size)

    polarity_test_dataloader = DataLoader(polarity_test_data_kelec, shuffle=True,
                                                  batch_size=batch_size)
    
    model = ElectraBaseClassifier_Cate_Base(len(tf_id_to_name), len(tokenizer_kelec))
    model.load_state_dict(torch.load(test_category_extraction_model_path_gpu_lr, map_location=device))
    model.to(device)
    model.eval()
            
    polarity_model = ElectraBaseClassifier_Pola_Base(len(polarity_id_to_name), len(tokenizer_kelec))
    polarity_model.load_state_dict(torch.load(test_polarity_classification_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form_kelec_forcing(tokenizer_kelec, model, polarity_model, copy.deepcopy(test_data_final))

    df_pred_final = pd.DataFrame(pred_data)

    df_final = pd.concat([df_pred_first, df_pred_second, df_pred_third, df_pred_fourth, df_pred_fifth, df_pred_final]).sort_values(by = ['id'], axis = 0).reset_index(drop = True)

    with open('/content/drive/MyDrive/Inference_samples.jsonl', 'w') as file:
        for i in range( len(df_final) ):
            if len(df_final['annotation'][i]) != 0 :
                tmp = str(df_final['annotation'][i]).replace("\'", "\"").replace('None', 'null')
                file.write(  '{'+'\"id\": \"{0}\", \"sentence_form\": \"{1}\", \"annotation\": {2}'\
                    .format( df_final['id'][i],   df_final['sentence_form'][i], tmp ) +'}' ) 
                file.write("\n")


    return pd.DataFrame(jsonlload('/content/drive/MyDrive/Inference_samples.jsonl'))
```
    
