from django.urls import path
from results import views

urlpatterns = [
    path('results/', views.snippet_list),
    path('results/<int:pk>/', views.snippet_detail),
]