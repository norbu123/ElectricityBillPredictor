from django.shortcuts import render 
import numpy as np
import pandas as pd
from joblib import load

from django.template import loader  
from django.http import HttpResponse 

model = load('./savedModels/model.joblib')

def predictor(request):
    if request.method=="POST":
        heating_days=float(request.POST['Heating_days'])
        cooling_days=float(request.POST['Cooling_days'])
        total_rooms=float(request.POST['TOTROOMS'])
        rooms_heated=float(request.POST['HEATROOM'])
        rooms_cooled=float(request.POST['ACROOMS'])
        total_etc=float(request.POST['Electricity_usage'])
        
        inputs=pd.DataFrame(data=({"Heating_days":[heating_days],"Cooling_days":[cooling_days],"TOTROOMS":[total_rooms], "HEATROOM":[rooms_heated],"ACROOMS":[rooms_cooled],"Electricity_usage":[total_etc]}))
        
        y_pred=model.predict(inputs)
        
        output = "Total bill predicted for a month: Nu  " + str(np.round(y_pred[0],4))

        return (render(request, 'main.html',{'result':output}))
    return render(request, 'main.html')

def homepage(request):  
   template = loader.get_template('homepage.html')
   return HttpResponse(template.render()) 
