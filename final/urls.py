from django.urls import path
from . import views

urlpatterns = [
    path('', views.final, name='final'),

]
