from django.contrib import admin

from .models import KerasImageNetClassifier


admin.site.register([
    KerasImageNetClassifier
])
