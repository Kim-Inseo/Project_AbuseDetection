# 필요 모듈 import
import os
import django
from django.utils.timezone import now
import pandas as pd
import time

# Django 환경 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AbuseDetection.settings')
django.setup()

# Django 모델 import
from app.models import SentenceInfo

def load_data():
    '''
    curse detection data(https://github.com/2runo/Curse-detection-data?tab=readme-ov-file)
    korean hate speech(https://github.com/kocohub/korean-hate-speech)
    을 각각 DB에 적재
    :return: None
    '''
    folder_path_txt = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    file_path_txt = os.path.join(folder_path_txt, 'dataset\dataset.txt')

    with open(file_path_txt, 'r', encoding='utf-8') as f:
        while True:
            row = f.readline() # 한 줄씩 읽는다.
            if not row: # 빈 줄인 경우
                break

            try:
                sentence, is_abuse = row.split('|')
                sentence = sentence.strip() # 양쪽 공백 제거
                is_abuse = int(is_abuse) # 0/1 값이므로 정수형으로 변환
            except: # 이 과정에서 오류가 나는 경우
                continue

            # 이미 존재하는 문장인지 확인
            if SentenceInfo.objects.filter(sentence=sentence).exists():
                print(f"[{sentence}]는 이미 처리된 데이터입니다.")
                continue  # 중복이면 건너뛴다

            # SentenceInfo 생성
            SentenceInfo.objects.create(
                data_source='curse detection data',
                sentence=sentence,
                is_abuse=is_abuse,
                timestamp=now(),
            )

            print(f"[{sentence}] 처리 완료")

    folder_path_tsv = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    file_path_tsv_train = os.path.join(folder_path_tsv, 'dataset\\train.tsv')
    file_path_tsv_dev = os.path.join(folder_path_tsv, 'dataset\dev.tsv')

    df_tsv_train = pd.read_csv(file_path_tsv_train, sep='\t')
    df_tsv_dev = pd.read_csv(file_path_tsv_dev, sep='\t')

    df_tsv = pd.concat([df_tsv_train, df_tsv_dev]).reset_index(drop=True) # 합쳐서 사용

    for _, row in df_tsv.iterrows(): # 매 행마다 반복
        comment = row['comments']
        comment = comment.strip() # 양쪽 공백 제고

        # 이미 존재하는 문장인지 확인
        if SentenceInfo.objects.filter(sentence=comment).exists():
            print(f"[{comment}]는 이미 처리된 데이터입니다.")
            continue  # 중복이면 건너뛴다

        if row['bias'] == 'none' and row['hate'] == 'none': # 차별성 발언, 혐오성 발언 모두 없는 경우
            is_abuse = 0 # 악성 글이 아니다.
            # SentenceInfo 생성
            SentenceInfo.objects.create(
                data_source='korean hate speech',
                sentence=comment,
                is_abuse=is_abuse,
                timestamp=now(),
            )
        else: # 하나라도 none이 아니면 (bias: gender, others, hate: hate, offensive)
            is_abuse = 1 # 악성 글이다.
            SentenceInfo.objects.create(
                data_source='korean hate speech',
                sentence=comment,
                is_abuse=is_abuse,
                timestamp=now(),
            )

        print(f"[{comment}] 처리 완료")

# 실행
if __name__ == '__main__':
    load_data()
    # --- 126.926 seconds ---

# CMD를 켜고 (venv) C:\Users\kis\Desktop\AbuseDetection>에서 set PYTHONPATH=%cd%
# 사전에 mysql 8.4 comment line client에서 create database abuse_detection_db default character set utf8 collate utf8_general_ci; 작성
# (venv) C:\Users\kis\Desktop\AbuseDetection>python app/scripts/data_loader.py