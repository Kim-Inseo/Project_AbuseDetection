# 필요 모듈 import
import os
import django
from tqdm import tqdm

# Django 환경 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AbuseDetection.settings')
django.setup()

# Django 모델 import
from app.models import SentenceInfo

# 전처리 모듈 import
import re
from kiwipiepy import Kiwi


# Code
def load_data_from_db():
    '''
    DB에서 텍스트를 불러오는 함수
    :return: sentence object(SentenceInfo)
    '''
    sentences = SentenceInfo.objects.all() # 모든 문장 불러오기
    for sentence in tqdm(sentences):
        yield sentence # 문장 반환

def clean_text(text):
    '''
    텍스트를 정제하는 함수.
    E-mail, URL, HTML, 특수문자, 기호를 제거한다.
    :param text: 정제 이전 텍스트
    :return text: 정제 이후 텍스트
    '''
    e_mail = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(e_mail, '', text) # E-mail 패턴 제거
    url = r'https?://\S+|www\.\S+'
    text = re.sub(url, '', text) # URL 패턴 제거
    html = r'<.*?>'
    text = re.sub(html, '', text) # HTML 패턴 제거
    mention = r'@\w+|@[가-힣]+'
    text = re.sub(mention, '', text)  # 멘션(@닉네임) 패턴 제거
    hashtag = r'#\w+|#[가-힣]+'
    text = re.sub(hashtag, '', text)  # 해시태그 패턴 제거

    return text

def normalize_text(text):
    '''
    1. 알파벳, 한글, 숫자, 문법 기호(!?.,"')가 아닌 것을 없애고,
    2. 반복되는 자음, 모음은 2번까지만 연속되도록 하고,
    3. ㅋ, ㅎ, ㅠ, ㅜ에 한해 서로 다른 자/모음은 띄도록 한다.
    :param text: normalize 이전 text
    :return text: normalize가 끝난 text
    '''
    text = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ0-9!?.,'\" ]", "", text)
    text = re.sub(r"([ㄱ-ㅎㅏ-ㅣ])\1{2,}", r"\1\1", text)
    text = re.sub(r"([ㅋㅎㅠㅜ])(?=(?!\1)[ㅋㅎㅠㅜ])", r"\1 ", text)
    return text

def tokenize(text):
    '''
    맞춤법과 띄어쓰기를 교정하고 토큰화하는 함수
    kiwipiepy를 이용한다.
    :param text: 교정 전 텍스트
    :return text: 교정 후 토큰화된 텍스트
    '''
    kiwi = Kiwi(
        model_type='sbg',  # 시간 절약을 위해서 오타 교정은 생략.
    )

    file_path = os.path.join(os.path.dirname(__file__), 'USER_DICT.dict')
    kiwi.load_user_dictionary(file_path)

    tokens = kiwi.tokenize(
        text,
        normalize_coda=True,  # ㅋㅋㅋ, ㅎㅎㅎ 같은 초성체가 어절 뒤에 붙는 경우를 고려
        z_coda=True,  # 조사나 어미에 덧붙은 받침을 분리
        split_complex=True,  # 더 잘게 분해 가능한 형태소들을 최대한 분할(파생 명사/동사/형용사)
    )

    # 사용자 등록 사전
    word_dict = []
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('#'):
                continue
            split_line = line.split('\t')
            word_dict.append(split_line[0])

    exclude_tag = [
        'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', # 조사
        'EP', 'EF', 'EC', 'ETN', 'ETM', # 어미
        'XSN', 'XSV', 'XSA', 'XSM', # 접미사
        'SF', 'SP', 'SS', 'SSO', 'SSC', 'SE', 'SO', 'SH', 'SB',
        # 기타 특수 문자(ㅋㅋ 등), 알파벳, 숫자를 제외한 부호, 외국어, 특수문자
        'UN', # 분석 불능
        'W_URL', 'W_EMAIL', 'W_HASHTAG', 'W_MENTION', 'W_SERIAL', 'W_EMOJI', # 웹
        # 사전에 제거한 부분도 있겠으나, 미처 제거되지 못한 게 있을 수 있으므로 추가 점검
        'Z_CODA', 'USER0', 'USER1', 'USER2', 'USER3', 'USER4', # 덧붙은 받침, 사용자 정의 태그
    ]

    after_exclude_tag = []

    # 기타 특수문자인데 사전에 등록되지 않은 경우는 생략
    # 제거해야 할 태그가 아닌 경우 등록
    for token in tokens:
        if token.tag == 'SW' and token.form not in word_dict:
            continue
        if token.tag not in exclude_tag:
            after_exclude_tag.append(token.form)

    text = ' '.join(after_exclude_tag)

    return text


def preprocess_and_save():
    '''
    DB에서 데이터를 불러와서 전처리 후 저장하는 함수.
    전처리가 완료된 문장은 새로운 열에 저장한다.
    :return: None
    '''
    for sentence in load_data_from_db():
        # 이미 처리된 메시지인지 확인
        # 처리에 장시간이 걸릴 것으로 예상되므로 중도에 끊을 수 있도록 하기 위해 작성
        if not isinstance(sentence.after_preprocessing, type(None)):
            print(f"해당 메시지 {sentence}는 이미 처리된 데이터입니다.")
            continue  # 중복이면 건너뛴다

        original_text = sentence.sentence # 원본 텍스트
        after_clean = clean_text(original_text)
        after_normalized = normalize_text(after_clean)
        preprocessed_text = tokenize(after_normalized) # 전처리 완료 텍스트

        # 전처리 후 텍스트를 DB의 새로운 필드에 저장
        sentence.after_preprocessing = preprocessed_text
        sentence.save()  # DB에 저장


# 실행
if __name__ == '__main__':
    preprocess_and_save()
    # 14187/14187 [8:40:00<00:00,  2.20s/it]


# CMD를 켜고 프로젝트 루트 디렉토리에서 set PYTHONPATH=%cd%
# (venv) C:\Users\kis\Desktop\AbuseDetection>python app/nlp/model_01/preprocess.py