from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

from main import views
from main.forms import AuthenticationForm
from satellite.views import satellite
from stocks.views import stocks

urlpatterns = [
    path(r'', views.index, name='index'),
    path(r'timeline/', views.timeline, name='timeline'),
    path(r'accounts/login/', LoginView.as_view(
        template_name='login.html',
        authentication_form=AuthenticationForm,
    ), name='login'),
    path(r'accounts/logout/', LogoutView.as_view(), name='logout'),
]
