from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload_csv/', views.upload_csv, name='upload_csv'),
]