# 필요 모듈 import
import os
import json
import django
import time

# Django 환경 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AbuseDetection.settings')
django.setup()

# pytorch 모듈
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        inputs = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {key: val.squeeze(0) for key, val in inputs.items()}


class AbuseDetector:
    def __init__(self):
        '''
        Transformer 모델, tokenizer 로드
        '''
        # 저장해두었던 것들 불러오기
        with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # train.py에서 정의한 모델 로드
        self.model_name = 'beomi/KcELECTRA-small-v2022'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(os.path.dirname(__file__), 'best_KcELECTRA_model.pth'),
                       map_location=self.device,
                       weights_only=True,
            )
        )
        self.threshold = config['threshold']

        self.model.eval()

    def detect_abuse(self, sentence_list):
        '''
        Transformer 모델을 사용한 악성 글 탐지
        :param comments: 판별할 텍스트 모음
        :return: True/False
        '''
        probs_list = []
        results = []

        dataset = InferenceDataset(sentence_list, tokenizer=self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=8)

        # 모델 예측
        with torch.no_grad():
            for batch in data_loader:
                inputs = {key: val.to(self.device) for key, val in batch.items()}
                output = self.model(**inputs)
                probs = F.softmax(output.logits, dim=1)[:, 1] # ex) tensor([0.0602, 0.9811, 0.9748], device='cuda:0')
                preds = (probs > self.threshold).long() # ex) tensor([0, 1, 1], device='cuda:0')

                probs_list += probs.tolist()
                results += ['ABUSE' if pred == 1 else 'NORMAL' for pred in preds] # 1이면 이상(True), 0이면 정상(False)

        return probs_list, results

# 인스턴스 생성
detector = AbuseDetector()

if __name__ == '__main__':
    start = time.time()

    sentence_list = [
        "애는 진짜 착한가보닼ㅋㅋㅋㅋㅋㅋ 친구도 쌍욕 박으면서 어찌어찌 상대는 해주는 거 보면ㅋㅋㅋㅋㅋㅋ",
        # 0.76866364 'ABUSE'
        "저런 친구 하나씩 있는데 저런 애들이 진짜 진국임. 멍청해도 의리 같은거 개쩔고 한편으로는 또 강아지 같아서 졸졸 잘 따라다니고. 딱 저런 개무식한 것만 잘 참을 수 있으면 저런 친구는 곁에 두는게 좋음. 그리고 저런 친구 곁에 있으려면 본인도 정말 괜찮은 놈이어야 하고.",
        # 0.9834499 'ABUSE'
        "컨셉이 아니라면 최소 목숨 구해준거 아니고서야 저렇게 친절하게 대화가 유지될 수가 없음 ㄹㅇㅋㅋ",
        # 0.08317665 'NORMAL'
        "것 저친구 진짜 웃겨서 숨 넘어갈뻔했네 ㅋㅋㅋㅋㅋㅋㅋ",
        # 0.3648776 'NORMAL'
        "진짜 ㅋㅋㅋㅋㅋ 존나 웃긴게 저 무식한 애 포함 3~4명 있을땐 항상 혼자만 못 웃고 주변 애들만 빵터질것 같아서 너무 불쌍하고 웃김 ㅋㅋ",
        # 0.98381823 'ABUSE'
        "ㅋㅋㅋㅋㅋㅋㅋㅋ모른다고는 안 하고 오답을 창의적으로 하네 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",
        # 0.082990535 'NORMAL'
        "'말을 안하면 완벽할 것 같은 사람'",
        # 0.07421104 'NORMAL'
        "것만 멀쩡한 친구 내 주변에도 있었으면 좋겠다 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",
        # 0.20724042 'NORMAL'
        "진짜 두 친구 다 착한 사람들인가봐요. 쌍욕하고 차단할법한데 꾸준히 대답해주는 친구나 욕먹어도 배움에 한계없이 납득하는 것(?)멀쩡한 친구나.",
        # 0.5320009 'ABUSE'
        '근데 "춘향이 아빠 장님" 하면 대부분의 사람들이 이상한거 눈치 못챌거같긴 함 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ',
        # 0.7135096 'ABUSE'
        ##
    ]

    probs_list, result = detector.detect_abuse(sentence_list)

    print(type(probs_list))
    print(type(probs_list[0]))
    print(probs_list)
    print(result)
    print('--- %.3f seconds ---' % (time.time() - start))
    # --- 2.218 seconds ---

# CMD를 켜고 프로젝트 루트 디렉토리에서 set PYTHONPATH=%cd%
# (venv) C:\Users\kis\Desktop\AbuseDetection>python app/nlp/model_02/detector.py