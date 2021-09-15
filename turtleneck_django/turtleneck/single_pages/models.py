from django.db import models



# Create your models here.
class User(models.Model):
    user_name = models.CharField(max_length=200, null=False)
    user_id = models.CharField(max_length=200, null=False)
    user_password = models.CharField(max_length=200, null=False)
    user_point = models.IntegerField(default=10000)
    def __str__(self):
        return self.user_name