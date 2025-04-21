# 필요 모듈 import
import os
import json
import django

import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Django 환경 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AbuseDetection.settings')
django.setup()

# Django 모델 import
from app.models import SentenceInfo

# vocab 구축 모듈
from collections import Counter

# padding 모듈
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Train 관련 모듈
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


# Code
def set_seed(seed: int=42):
    '''
    :param seed: 고정할 시드 값
    :return: None
    '''
    random.seed(seed) # python 내장 random 모듈 seed
    np.random.seed(seed) # numpy seed
    torch.manual_seed(seed) # pytorch seed
    torch.cuda.manual_seed(seed) # cuda 연산 seed
    torch.cuda.manual_seed_all(seed) # 멀티 GPU seed
    torch.backends.cudnn.deterministic = True # CuDNN 연산을 결정론적으로 설정
    torch.backends.cudnn.benchmark = False # 연산 최적화를 비활성화(일관된 결과 보장)

def load_data_from_db():
    '''
    QuerySet을 이용해 DataFrame 구축
    :return df: pandas dataframe
    '''
    queryset = SentenceInfo.objects.all().values()
    df = pd.DataFrame(list(queryset))

    df.drop(['id', 'timestamp'], axis=1, inplace=True)

    return df

def split_data(df):
    '''
    원본 데이터프레임을 분석에 맞게 변환하는 과정
    :param df: 분리 전 데이터프레임
    :return X_train, X_val, X_test, y_train, y_val, y_test: pandas dataframe
    '''
    # X, y 분할
    X, y = df.drop(['is_abuse'], axis=1), df.is_abuse

    # data source에 따라 분할
    X_curse, y_curse = X[X['data_source'] == 'curse detection data'], y[X['data_source'] == 'curse detection data']
    X_hate, y_hate = X[X['data_source'] == 'korean hate speech'], y[X['data_source'] == 'korean hate speech']

    # curse detection data에 대해 train, val, test 분할
    X_curse_temp, X_curse_test, y_curse_temp, y_curse_test = train_test_split(
        X_curse, y_curse, test_size=0.1, stratify=y_curse, random_state=42,
    )
    X_curse_train, X_curse_val, y_curse_train, y_curse_val = train_test_split(
        X_curse_temp, y_curse_temp, test_size=1/9, stratify=y_curse_temp, random_state=42,
    ) # 0.9 * 1/9 = 0.1

    # korean_hate_speech에 대해 train, val, test 분할
    X_hate_temp, X_hate_test, y_hate_temp, y_hate_test = train_test_split(
        X_hate, y_hate, test_size=0.1, stratify=y_hate, random_state=42,
    )
    X_hate_train, X_hate_val, y_hate_train, y_hate_val = train_test_split(
        X_hate_temp, y_hate_temp, test_size=1/9, stratify=y_hate_temp, random_state=42,
    ) # 0.9 * 1/9 = 0.1

    # 합치기
    X_train = pd.concat([X_curse_train, X_hate_train])
    y_train = pd.concat([y_curse_train, y_hate_train])
    X_val = pd.concat([X_curse_val, X_hate_val])
    y_val = pd.concat([y_curse_val, y_hate_val])
    X_test = pd.concat([X_curse_test, X_hate_test])
    y_test = pd.concat([y_curse_test, y_hate_test])

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_vocab(df):
    '''
    :param df: dataframe of training data
    :return: vocabulary
    '''
    tokens_list = df.after_preprocessing.values

    # 모든 문장에서 단어의 빈도수 계산
    word_counter = Counter()
    for tokens in tokens_list:
        word_counter.update(tokens.split())

    # 단어를 빈도 순으로 정렬해 인덱스 부여
    vocab = {'<pad>': 0, '<unk>': 1} # special tokens
    vocab.update({word: idx+2 for idx, (word, _) in enumerate(word_counter.most_common())})

    return vocab

def word_to_index(vocab, word):
    '''
    :param vocab: 단어 사진
    :param word: 변환할 단어
    :return: 인덱스 값
    '''
    return vocab.get(word, vocab['<unk>']) # OOV 단어는 <unk>로 치환

def encode_sentence(vocab, sentence):
    '''
    :param vocab: 단어 사전
    :param sentence: 변환할 문장
    :return: encoded sentence tensor
    '''
    if len(sentence) == 0: # 빈 문장인 경우
        return [vocab['<pad>']]
    else:
        return torch.tensor([word_to_index(vocab, word) for word in sentence.split()])


def padding(X_train_encoded, X_val_encoded, X_test_encoded, vocab):
    '''
    train 데이터의 최대 길이의 2배를 기준점으로 설정하여 padding 과정 진행
    :param X_train_encoded: 인코딩 완료된 train data
    :param X_val_encoded: 인코딩 완료된 val data
    :param X_test_encoded: 인코딩 완료된 test data
    :param vocab: 단어 사전
    :return: X_train_padded, X_val_padded, X_test_padded
    '''
    # train 데이터 최대 길이의 2배를 max_length 설정
    max_length = max(len(seq) for seq in X_train_encoded) * 2

    # TF pad_sequences
    X_train_padded = pad_sequences(
        X_train_encoded, maxlen=max_length, padding='post', truncating='post', value=vocab['<pad>'],
    )
    X_val_padded = pad_sequences(
        X_val_encoded, maxlen=max_length, padding='post', truncating='post', value=vocab['<pad>'],
    )
    X_test_padded = pad_sequences(
        X_test_encoded, maxlen=max_length, padding='post', truncating='post', value=vocab['<pad>'],
    )

    # TF -> NumPy -> Pytorch
    X_train_padded = torch.tensor(np.array(X_train_padded), dtype=torch.long)
    X_val_padded = torch.tensor(np.array(X_val_padded), dtype=torch.long)
    X_test_padded = torch.tensor(np.array(X_test_padded), dtype=torch.long)

    return X_train_padded, X_val_padded, X_test_padded


## Transformer 모델 정의
class TransformerModel(nn.Module):
    '''
    vocab_size: vocab 크기
    d_model: embedding 차원
    nhead: multi-head attention head 개수
    num_encoder_layers: Transformer encoder의 층수
    num_classes: 클래스 개수
    '''
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, hidden_dim, num_classes):
        super(TransformerModel, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=num_encoder_layers,
        )

        # binary classification을 위한 output layer (2 classes)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x) # [batch_size, seq_len, d_model]
        transformer_out = self.transformer_encoder(embedded) # [batch_size, seq_len, d_model]
        output = transformer_out.mean(dim=1) # [batch_size, d_model], 평균 풀링 사용
        output = self.fc(output) # [batch_size, num_classes]

        return output


## train, val, test
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, patience=3):
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # tqdm으로 진행 상황 표시
        for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation 평가 (매 epoch마다)
        val_f1 = evaluate_model(model, val_loader, device=device) # 기본 threshold는 0.5
        print(f'Epoch {epoch+1}: Loss: {running_loss:.4f}, Validation F1: {val_f1:.4f}')

        # Early stopping 체크하기
        if val_f1 > (best_f1 + 0.001): # 최소치 이상 상승하지 않으면 개선되지 않은 것으로 본다.
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'best_model.pth'))
            print('성능이 개선되어 모델을 저장합니다.')
        else:
            patience_counter += 1
            print(f'성능이 개선되지 않았습니다. 조기 종료까지 {patience_counter}/{patience}')
            if patience_counter >= patience: # 조기 종류 규칙에 도달하면
                print("조기 종료 조건을 만족하여 학습을 종료합니다.")
                break

    print(f"\n Best Validation F1: {best_f1:.4f} at Epoch {best_epoch+1}.")

def evaluate_model(model, val_loader, threshold=0.5, device=None):
    model.eval()
    all_preds = []
    all_labels = []

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] >= threshold).long()  # class 1 기준 threshold 이상일 때만 1을 반환

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro') # F1-score (macro average)

        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        return f1

def find_best_threshold(model, val_loader, device=None):
    model.eval()
    all_probs = []
    all_labels = []

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    thresholds = [i * 0.05 for i in range(6, 11)] # 재현율에 우선순위를 둠.
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds = [1 if p >= t else 0 for p in all_probs]
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
        print(f"Threshold {t:.2f} F1: {best_f1:.4f}")

    print(f"\n Best Threshold {best_threshold:.2f} with F1: {best_f1:.4f}")
    return best_threshold



# 실행
if __name__ == '__main__':
    set_seed(42)

    df = load_data_from_db()
    df.to_csv(os.path.join(os.path.dirname(__file__), 'preprocessed_df.csv'), index=False, encoding='utf-8-sig')
    # 전처리 반복 작업 방지를 위한 CSV 파일 저장

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    vocab = make_vocab(X_train)  # train data 기준으로 vocabulary 생성

    X_train_encoded = [encode_sentence(vocab, sentence) for sentence in X_train.after_preprocessing.values]
    X_val_encoded = [encode_sentence(vocab, sentence) for sentence in X_val.after_preprocessing.values]
    X_test_encoded = [encode_sentence(vocab, sentence) for sentence in X_test.after_preprocessing.values]

    X_train_padded, X_val_padded, X_test_padded = padding(X_train_encoded, X_val_encoded, X_test_encoded, vocab)

    # DataLoader 설정
    train_set = TensorDataset(X_train_padded, torch.tensor(y_train.values))
    val_set = TensorDataset(X_val_padded, torch.tensor(y_val.values))
    test_set = TensorDataset(X_test_padded, torch.tensor(y_test.values))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # 모델 instance화
    vocab_size = len(vocab)  # vocab 크기
    d_model = 128  # embedding 차원, if error, 256 -> 64
    nhead = 4  # multi-head attention head 수, if error, 8 -> 4
    num_encoder_layers = 3  # transformer encoder 층수, if error, 6 -> 3
    hidden_dim = 256 # Feed-Forward 네트워크의 hidden dimension, if error, 512 -> 128
    num_classes = 2  # binary classification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, hidden_dim, num_classes).to(device)

    # loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # 모델 학습 (Early stopping + best model 저장)
    train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=30) # Best Validation F1: 0.7165 at Epoch 3.

    # best model 로드
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'best_model.pth'),
                                     weights_only=True))

    # 최적 threshold 탐색
    best_threshold = find_best_threshold(model, val_loader, device) # Best Threshold 0.30 with F1: 0.7205

    # test 성능 평가
    test_f1 = evaluate_model(model, test_loader, threshold=best_threshold, device=device)
    print(f"\n Test F1 (Best Threshold={best_threshold:.2f}): {test_f1:.4f}") # Test F1 (Best Threshold=0.30): 0.6912

    # json 파일로 vocab, padding 길이, 모델 설정 저장
    config = {
        'vocab': vocab,
        'pad_length': X_train_padded.shape[1],
        'params': {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'hidden_dim': hidden_dim,
            'num_classes': num_classes,
        },
        'threshold': best_threshold,
    }
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# CMD를 켜고 프로젝트 루트 디렉토리에서 set PYTHONPATH=%cd%
# (venv) C:\Users\kis\Desktop\AbuseDetection>python app/nlp/model_01/train.py
