from django.urls import path

from courses import views

urlpatterns = [
    path(r'', views.courses, name='courses'),
    path(r'week1', views.week1, name='week1'),
    path(r'week2', views.week2, name='week2'),
    path(r'exercise_2_3', views.exercise_2_3, name='exercise_2_3'),
    path(r'exercise_3_1', views.exercise_3_1, name='exercise_3_1'),
    path(r'exercise_3_4', views.exercise_3_4, name='exercise_3_4'),
    path(r'assignment_1', views.assignment_1, name='assignment_1'),
    path(r'assignment_1_data', views.assignment_1_data, name='assignment_1_data'),
    path(r'assignment_1_tikh', views.assignment_1_tikh, name='assignment_1_tikh'),
    path(r'assignment_1_knee', views.assignment_1_knee, name='assignment_1_knee'),
    path(r'assignment_1_disc', views.assignment_1_disc, name='assignment_1_disc'),
    path(r'assignment_1_gcv', views.assignment_1_gcv, name='assignment_1_gcv'),
    path(r'assignment_1_full', views.assignment_1_full, name='assignment_1_full'),

]
