from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('predictor', views.predictor, name='predictor'),
    

]