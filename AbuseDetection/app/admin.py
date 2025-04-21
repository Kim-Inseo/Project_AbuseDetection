from django.contrib import admin
from app.models import PredictionLog

# Register your models here.
@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "predict", "probability", "text")
    list_filter = ("predict", "timestamp")
    search_fields = ("text",)