from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    return render(request, 'homepage.html')

def user(request):
    username = request.GET['username']
    return render(request, 'user.html', {'name':username})