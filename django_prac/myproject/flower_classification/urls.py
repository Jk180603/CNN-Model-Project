from django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_flower, name='classify_flower'),
]