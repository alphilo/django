from django.shortcuts import render

# Create your views here.
from .models import DataPoint
import matplotlib.pyplot as plt
import numpy as np

def plot_data(request):
    data = DataPoint.objects.all()
    x_values = [point.x_value for point in data]
    y_values = [point.y_value for point in data]

    plt.scatter(x_values, y_values)
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.title('Data Visualization')
    plt.grid(True)

    # Save plot to a temporary file
    plot_path = 'data_visualization/static/plot.png'
    plt.savefig(plot_path)
    plot_path = '../../static/plot.png';

    return render(request, 'data_visualization/plot.html', {'plot_path': plot_path})