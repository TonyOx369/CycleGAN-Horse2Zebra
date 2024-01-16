
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    path('', views.main_page, name='main_page'),
    path('translate/<str:option>/<int:has_verification>/', views.translation_page, name='translation'),
]
