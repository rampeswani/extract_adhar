from django.contrib import admin
from django.urls import path,include
from .views import *
urlpatterns = [
    
    path('process-aadhaar/', process_aadhaar_image, name='process_aadhaar'),
    path('test',extract_text_from_image , name='test'),
    path('new-api/',OCR , name='ocr')
]
