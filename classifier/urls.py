from django.urls import path
from .views import ClassifierView, DefaultView

app_name = "classifier"

urlpatterns = [
    path('classify/', ClassifierView.as_view()),
    path('', DefaultView.as_view()),
]