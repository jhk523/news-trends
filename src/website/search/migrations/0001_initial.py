# Generated by Django 2.1 on 2020-02-20 07:30

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='NewsArticle',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateTimeField(blank=True, null=True)),
                ('publisher', models.CharField(max_length=255)),
                ('title', models.CharField(max_length=255)),
                ('author', models.CharField(blank=True, max_length=255, null=True)),
                ('link', models.CharField(blank=True, max_length=255, null=True, unique=True)),
                ('description', models.TextField(blank=True, null=True)),
            ],
        ),
    ]
