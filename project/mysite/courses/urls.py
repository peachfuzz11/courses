from django.urls import path

from courses import views

urlpatterns = [
    path(r'', views.courses, name='courses'),
    path(r'week1', views.week1, name='week1'),
    path(r'week2', views.week2, name='week2'),
    path(r'exercise_2_3', views.exercise_2_3, name='exercise_2_3'),
    path(r'exercise_3_1', views.exercise_3_1, name='exercise_3_1'),

]
