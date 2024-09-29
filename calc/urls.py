from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('requestDemo/', views.request_demo, name='request_demo'), 
    path('dashboard/', views.dashboard, name='dashboard_'), 
    path('requestDemo/',views.request_demo,name='requestDemo'),
    path('image-comparison/', views.image_comparsion_view, name='image_comparison'),  
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
