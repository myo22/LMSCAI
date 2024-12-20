import numpy as np
import pandas as pd
from dezero import optimizers
from models.siamese_network import SiameseNetwork
import dezero.functions as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# BERT 임베딩 함수
def embed_sentence(sentence):
    if not isinstance(sentence, str):  # sentence가 문자열이 아닌 경우 빈 문자열로 처리
        sentence = ""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    cls_embedding = last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.astype(np.float32)

# 데이터 준비 (sts-train.tsv 파일을 로드)
df = pd.read_csv('sts-train.tsv', sep='\t', on_bad_lines='skip')

# NaN 값 처리 및 필요한 컬럼 정의
df['sentence1'] = df['sentence1'].fillna("")  # NaN 값을 빈 문자열로 대체
df['sentence2'] = df['sentence2'].fillna("")  # NaN 값을 빈 문자열로 대체

# 데이터셋에서 댓글 벡터화
X1 = np.array([embed_sentence(sentence) for sentence in df['sentence1']])
X2 = np.array([embed_sentence(sentence) for sentence in df['sentence2']])

# 라벨 값
y = df['score'].values.astype(np.float32)

# 모델 초기화
input_dim = X1.shape[1]  # 벡터의 차원 (768)
model = SiameseNetwork(input_dim=input_dim)

# 옵티마이저 설정
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)

# 학습
n_epochs = 100
for epoch in range(n_epochs):
    loss_accum = 0
    for i in range(len(y)):
        x1, x2, label = X1[i:i+1], X2[i:i+1], y[i:i+1]

        # SiameseNetwork을 통해 예측값 계산
        pred = model(x1, x2)
        
        # MSE (Mean Squared Error)를 사용하여 유사도 예측
        loss = F.mean_squared_error(pred, label)
        
        model.cleargrads()  # 기울기 초기화
        loss.backward()  # 역전파
        optimizer.update()  # 최적화
        
        loss_accum += loss.data
        
    print(f"Epoch {epoch+1}, Loss: {loss_accum / len(y)}")

# 모델 저장
model.save_weights("siamese_model_weights.npz")