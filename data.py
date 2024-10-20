import json
import torch
from transformers import GPT2Model, GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset

# GPU/CPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################

# SST_train.json 파일 로드
with open('../dataset/sa/SST-2/SST_train.json', 'r', encoding='utf-8') as f:
    sst_data = json.load(f)

# # NSMC_train.json 파일 로드
# with open('../dataset/sa/NSMC/NSMC_train.json', 'r', encoding='utf-8') as f:
#     nsmc_data = json.load(f)

# SST 데이터셋에서 텍스트와 레이블 추출
sst_texts = [item['review'] for item in sst_data]
sst_labels = [item['sentiment'] for item in sst_data]

# # NSMC 데이터셋에서 텍스트와 레이블 추출
# nsmc_texts = [item['document'] for item in nsmc_data]
# nsmc_labels = [item['label'] for item in nsmc_data]

#######################################################

# GPT-2 모델 및 토크나이저 로드
gpt2_model = GPT2Model.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 패딩 토큰 설정 (eos_token을 pad_token으로 사용)
tokenizer.pad_token = tokenizer.eos_token

# SST, NSMC tokenization
sst_tokenized_texts = tokenizer(sst_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
# nsmc_tokenized_texts = tokenizer(nsmc_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
print("The tokenization has been successfully completed.")

# TensorDataset 생성
sst_dataset = TensorDataset(sst_tokenized_texts['input_ids'], sst_tokenized_texts['attention_mask'])

# DataLoader로 batch size 설정
# - batch size를 명시적으로 설정하지 않으면, SST 데이터셋 전체를 한 번에 토큰화하고, 이를 모델에 입력으로 사용하게 됨
# - (batch size: 모델이 한 번에 처리하는 입력 데이터의 개수)
batch_size = 16
sst_loader = DataLoader(sst_dataset, batch_size=batch_size)

# batch 단위로 GPT-2 모델에 입력
sst_embeddings = []
with torch.no_grad():
    for input_ids, attention_mask in sst_loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        sst_embeddings.append(outputs.last_hidden_state)
    print("Embedding vector extraction completed.")
