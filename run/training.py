import sys
import os
import torch
from transformers import GPT2Model

# 현재 파일의 경로 기준으로 상위 디렉터리(RNN 폴더 경로) 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# data.py 모듈을 가져오기
import data

# GPT-2 모델 로드
gpt2_model = GPT2Model.from_pretrained('gpt2')

# GPT-2 모델의 임베딩 레이어를 통해 임베딩 벡터 추출
with torch.no_grad():
    sst_embeddings = gpt2_model(**data.sst_tokenized_texts).last_hidden_state
    nsmc_embeddings = gpt2_model(**data.nsmc_tokenized_texts).last_hidden_state
    print("It passed through the embedding layer.")