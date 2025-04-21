### 라우팅 설정

from django.urls import path
from .views import index, check_comment, check_comments
from . import views # views.py가 있는 폴더 기준

urlpatterns = [
    path('', index, name='index'),
    path('check_comment/', check_comment, name='check_comment'),
    path('check_comments/', check_comments, name='check_comments'),
]