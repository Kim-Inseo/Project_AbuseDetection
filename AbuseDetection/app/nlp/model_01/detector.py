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
from app.nlp.model_01.preprocess import clean_text, normalize_text, tokenize
from app.nlp.model_01.train import word_to_index, encode_sentence, TransformerModel

# TF 모듈
from tensorflow.keras.preprocessing.sequence import pad_sequences

class AbuseDetector:
    def __init__(self):
        '''
        Transformer 모델, tokenizer 로드
        '''
        # 저장해두었던 것들 불러오기
        with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.vocab = config['vocab']
            self.pad_length = config['pad_length']
            self.params = config['params']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # train.py에서 정의한 모델 로드
        self.model = TransformerModel(**self.params).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(os.path.dirname(__file__), 'best_model.pth'),
                       map_location=self.device,
                       weights_only=True,
            )
        )
        self.threshold = config['threshold']
        self.model.eval()

    def detect_abuse(self, comments):
        '''
        Transformer 모델을 사용한 악성 글 탐지
        :param comments: 판별할 텍스트 모음
        :return: True/False
        '''
        results = []

        # 전처리
        after_clean = [clean_text(comment) for comment in comments]
        after_normalized = [normalize_text(text) for text in after_clean]
        preprocessed_text = [tokenize(text) for text in after_normalized]

        # encoding, padding, tensor로 만들기
        text_encoded = [encode_sentence(self.vocab, text) for text in preprocessed_text]
        text_padded = pad_sequences(
            text_encoded, maxlen=self.pad_length, padding='post', truncating='post', value=self.vocab['<pad>'],
        )
        text_tensor = torch.tensor(text_padded).to(self.device)

        # 모델 예측
        with torch.no_grad():
            logits = self.model(text_tensor) # ex) tensor([[ 0.5559, -0.9167]], device='cuda:0')
            probs = F.softmax(logits, dim=1) # ex) tensor([[0.8135, 0.1865]], device='cuda:0')
            preds = (probs[:, 1] >= self.threshold).long() # ex) [0]

            results = ['ABUSE' if pred == 1 else 'NORMAL' for pred in preds] # 1이면 이상(True), 0이면 정상(False)

            print(probs)
            print(preds)
            print(results)

        return results

# 인스턴스 생성
detector = AbuseDetector()

if __name__ == '__main__':
    start = time.time()

    sentence_list = [
        "이거 너무 좋은데요!",
        "XX놈아 꺼져!",
        "저런 친구 하나씩 있는데 저런 애들이 진짜 진국임. 멍청해도 의리 같은거 개쩔고 한편으로는 또 강아지 같아서 졸졸 잘 따라다니고. 딱 저런 개무식한 것만 잘 참을 수 있으면 저런 친구는 곁에 두는게 좋음. 그리고 저런 친구 곁에 있으려면 본인도 정말 괜찮은 놈이어야 하고.",
        "밥먹으면서 마지막거보다가 뿜었어ㅠㅠㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",
        "무료레슨 10만원은 진짜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 레슨은 자기가 받아야 될 거 같은데ㅋㅋㅋㅋㅋㅋ",
        "144헤르츠 빌런 저새끼는 액정깨진 폰으로 화면 캡쳐하면 깨진 화면 나오는줄 아는거 아님?ㅋㅋ",
        "인간은 자기 상식수준에 벗어나는 말을 들으면 어떻게든 반박을 하고 믿지 않을려고하죠.",
        "저기 그 여름에 냉장고에 물 넣으니까 시원해졌다는 분은 어투보니까 걍 더위먹으신 분같음 ㅋㅋㅋㅋㅋ",
        "모르면 알아가면 문제가 없다 하지만 알려줘도 못알아 먹으면 문제가 있다",
        "와.. 저런 사람들은 그동안 세상을 어떻게 산 거지? 그리고 젤 빡침 포인트는 많은 사람들이 알려주는데도 자기가 맞다고 벅벅 우기는 거ㅋㅋㅋ 진짜 극혐 이다",
        "이세계도 거부한 지능 드립은 진짜 미친거냐 ㅋㅋㅋㅋ"
    ]

    result = detector.detect_abuse(sentence_list)

    print('--- %.3f seconds ---' % (time.time() - start))
    # --- 24.155 seconds ---

# CMD를 켜고 프로젝트 루트 디렉토리에서 set PYTHONPATH=%cd%
# (venv) C:\Users\kis\Desktop\AbuseDetection>python app/nlp/model_01/detector.py