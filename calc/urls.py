from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('add', views.add, name='add'),
    path('upload', views.upload, name='upload'),
    path('check', views.check, name='check'),
    path('mid', views.mid, name='mid'),
    path('detail', views.detail, name='detail'),
]
