# Generated by Django 3.1.13 on 2021-07-21 22:45


from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('AI', '0001_keras_imagenet_classifier')
    ]

    operations = [
        migrations.AlterModelOptions(
            name='kerasimagenetclassifier',
            options={'base_manager_name': 'objects',
                     'verbose_name': 'Keras ImageNet Classifier',
                     'verbose_name_plural': 'Keras ImageNet Classifiers'})
    ]
