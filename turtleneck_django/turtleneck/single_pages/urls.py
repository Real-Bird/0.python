
from django.urls import path
from django.urls.resolvers import URLPattern
from . import views


urlpatterns = [
  
    path('about_me/', views.about_me),
    path('',views.landing),
    path('login/', views.loginview, name='loginview'),
    path('loginprocess/', views.loginprocess, name='loginprocess'),
    path('signup/', views.signup, name='signup'),
    path('', views.landing, name='home'),
    path('logout/', views.logout, name='logout'),
    path('forw_point/', views.forw_point, name='forwpoint'),
    path('corr_point/', views.corr_point, name='corrpoint'),
    path('view_point/', views.view_point, name='viewpoint'),
]