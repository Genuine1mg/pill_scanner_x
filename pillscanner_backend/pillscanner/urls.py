from django.urls import path
from .views import ExtractMedicineName

urlpatterns = [
    path('extract_medicine_name/', ExtractMedicineName.as_view(), name='extract_medicine_name')
]
