from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

# Create your models here.
class SentenceInfo(models.Model):
    data_source = models.TextField()
    sentence = models.TextField()
    is_abuse = models.IntegerField(validators=[
        MinValueValidator(0),
        MaxValueValidator(1)
    ])
    timestamp = models.DateTimeField(auto_now_add=True)

    # 전처리 후 텍스트 저장
    after_preprocessing = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.sentence


class PredictionLog(models.Model):
    uuid = models.CharField(max_length=100, unique=True)
    text = models.TextField()
    probability = models.FloatField()
    predict = models.CharField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.uuid} - {self.text} - {self.predict}"


# mysql> create database abuse_detection_db default character set utf8 collate utf8_general_ci;