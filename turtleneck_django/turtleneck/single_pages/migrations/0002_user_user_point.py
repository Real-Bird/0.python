# Generated by Django 3.2.5 on 2021-09-07 16:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('single_pages', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='user_point',
            field=models.IntegerField(default=10000),
        ),
    ]
