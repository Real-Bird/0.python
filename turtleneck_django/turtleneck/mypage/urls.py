

from django.urls import path

from . import views
urlpatterns = [
    path('', views.mypage),
]

# urlpatterns = [
#     path('search/<str:q>/', views.MyPostSearch.as_view()),

#     path('delete_mypost/<int:pk>/',views.delete_mypost),
  
#     path('update_mypost/<int:pk>/', views.MyPostUpdate.as_view()),
#     path('create_mypost/', views.MyPostCreate.as_view()),
 
#     path('tag/<str:slug>/', views.tag_page),
#     path('category/<str:slug>/', views.category_page),

#     path('<int:pk>/', views.MyPostDetail.as_view()),
#     path('', views.MyPostList.as_view()),

   
#     #path('', views.index),
#     #path('<int:pk>/',views.single_post_page ),
  
# ]