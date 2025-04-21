### Django 뷰
import json
import traceback
from django.http import JsonResponse
from app.nlp.model_02.detector import detector
from app.models import PredictionLog

# Local
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

# Chrome
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
## Local
@ensure_csrf_cookie
def index(request):
    return render(request, 'index.html')

def check_comment(request):
    # 댓글을 받아 악성 여부 판별
    if request.method == "POST":
        comment = request.POST.get('comment', '')

        prob, pred = detector.detect_abuse([comment])
        print(f"Predict: '{comment}' => {pred[0]} ({prob[0]:.4f})")

        return JsonResponse({'comment': comment, 'is_abuse': pred[0], 'probability': prob[0]})

    return JsonResponse({'error': 'Invalid request'}, status=400)

## Chrome
def save_prediction(uuid, text, prob, pred):
    if not PredictionLog.objects.filter(uuid=uuid).exists():
        PredictionLog.objects.create(
            uuid=uuid,
            text=text,
            probability=prob,
            predict=pred
        )

@csrf_exempt
def check_comments(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode('utf-8'))
            comment_objs = data # [{id: uuid, text: '...'}, ...]

            texts = [c["text"] for c in comment_objs]
            ids = [c["id"] for c in comment_objs]

            probs_list, preds = detector.detect_abuse(texts)

            response_data = []
            for uuid, text, prob, pred in zip(ids, texts, probs_list, preds):
                print(f"Predict: '{text}' => {pred} ({prob:.4f})")
                save_prediction(uuid, text, prob, pred)
                response_data.append({
                    "id": uuid,
                    "prob": prob,
                    "predict": pred
                })

            return JsonResponse(response_data, safe=False)

        except Exception as e:
            print("=== Error Traceback ===")
            print(traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({"error": "Invalid method"}, status=405)
