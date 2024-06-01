from django.urls import path
from .views import plot_data
from .fakenewsviews import fake_news
from .machinelearning import machine_learning

urlpatterns = [
    path('plot/', plot_data, name='plot_data'),
    path('fakenews/', fake_news, name='fake_news'),
    path('machinelearning/', machine_learning, name='machine_learning')
]
