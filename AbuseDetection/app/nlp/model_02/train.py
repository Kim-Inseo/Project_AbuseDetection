# 필요 모듈 import
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import django
import numpy as np
import pandas as pd
import random
import json

# Django 환경 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AbuseDetection.settings')
django.setup()

# Django 모델 import
from app.models import SentenceInfo

# 전처리 모듈 import
import re
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Train 관련 모듈
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm

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

    df = df[['data_source', 'sentence', 'is_abuse']]

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

# 데이터셋 클래스
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


## train, val, test
def train_model(model, train_loader, val_loader, optimizer, num_epochs=5, patience=2):
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # tqdm으로 진행 상황 표시
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in loop:
            batch = {key: val.to(model.device) for key, val in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

        # Validation 평가 (매 epoch마다)
        val_f1 = evaluate_model(model, val_loader) # 기본 threshold는 0.5
        print(f'Epoch {epoch+1}: Loss: {total_loss:.4f}, Validation F1: {val_f1:.4f}')

        # Early stopping 체크하기
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'best_KcELECTRA_model.pth'))
            print('성능이 개선되어 모델을 저장합니다.')
        else:
            patience_counter += 1
            print(f'성능이 개선되지 않았습니다. 조기 종료까지 {patience_counter}/{patience}')
            if patience_counter >= patience: # 조기 종류 규칙에 도달하면
                print("조기 종료 조건을 만족하여 학습을 종료합니다.")
                break

def evaluate_model(model, val_loader, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {key: val.to(model.device) for key, val in batch.items()}

            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1] # Positive class
            all_preds += (probs > threshold).cpu().numpy().tolist()  # class 1 기준 threshold 이상일 때만 1을 반환
            all_labels += batch['labels'].cpu().numpy().tolist()

        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds) # F1-score

        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        return f1

def find_best_threshold(model, val_loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]  # Positive class

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    # 다양한 threshold 평가
    best_f1 = 0.0
    best_threshold = 0.5

    for t in np.arange(0.1, 0.9, 0.01):
        preds = (np.array(all_probs) > t).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
        print(f"Threshold {t:.2f} F1: {f1:.4f}")

    print(f"\n Best Threshold {best_threshold:.2f} with F1: {best_f1:.4f}")
    return best_threshold


# 실행
if __name__ == '__main__':
    set_seed(42)

    # 데이터 불러오기
    df = load_data_from_db()

    # 데이터 분할
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_train_text = X_train['sentence'].tolist()
    X_val_text = X_val['sentence'].tolist()
    X_test_text = X_test['sentence'].tolist()
    y_train_label = y_train.tolist()
    y_val_label = y_val.tolist()
    y_test_label = y_test.tolist()

    # KcELECTRA-small-v2022 토크나이저 로딩
    model_name = 'beomi/KcELECTRA-small-v2022'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # TensorDataset 생성
    train_dataset = TextDataset(X_train_text, y_train_label, tokenizer)
    val_dataset = TextDataset(X_val_text, y_val_label, tokenizer)
    test_dataset = TextDataset(X_test_text, y_test_label, tokenizer)

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 모델 학습 (Early stopping + best model 저장)
    train_model(model, train_loader, val_loader, optimizer, num_epochs=10, patience=2) # Loss: 231.4857, Validation F1: 0.8138

    # best model 로드
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'best_KcELECTRA_model.pth'),
                                     weights_only=True))
    model.to(device)

    # 검증 dataset에서 최적 threshold 탐색
    best_threshold = find_best_threshold(model, val_loader) # Best Threshold 0.42 with F1: 0.8155

    # test에서 최적의 threshold 적용하여 성능 평가
    test_f1 = evaluate_model(model, test_loader, threshold=best_threshold)
    print(f"\n Test F1 (Best Threshold={best_threshold:.2f}): {test_f1:.4f}")
    # Precision: 0.7250 | Recall: 0.8598 | F1: 0.7866
    # Test F1 (Best Threshold=0.42): 0.7866

    # json 파일로 vocab, padding 길이, 모델 설정 저장
    config = {
        'threshold': best_threshold,
    }
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# CMD를 켜고 프로젝트 루트 디렉토리에서 set PYTHONPATH=%cd%
# (venv) C:\Users\kis\Desktop\AbuseDetection>python app/nlp/model_02/train.py

