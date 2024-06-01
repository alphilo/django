from django.db import models

# Create your models here.

class Student(models.Model):
    x_value = models.FloatField()
    y_value = models.FloatField()
    
class DataPoint(models.Model):
    x_value = models.FloatField()
    y_value = models.FloatField()