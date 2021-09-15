from django.urls import path
from .import views

urlpatterns = [
    path('update_notice/<int:pk>/', views.NoticeUpdate.as_view()),
    # path('create_notice/', views.NoticeCreate.as_view()),
    path('category/<str:slug>/', views.category_page),
    path('<int:pk>/', views.NoticeDetail.as_view()),
    path('', views.NoticeList.as_view()),
]
