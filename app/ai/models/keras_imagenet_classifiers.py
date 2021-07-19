from h1st.django.model.api import H1stModel

from django.db.models.fields import CharField

from tensorflow.keras.applications.densenet import (
    DenseNet121,
    DenseNet169,
    DenseNet201
)
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception


class KerasImageNetClassifier(H1stModel):
    name = CharField(
            max_length=255,
            unique=True)

    class Meta:
        verbose_name = "Keras ImageNet Classifier"
        verbose_name_plural = "Keras ImageNet Classifiers"

    def __str__(self) -> str:
        return f'"{self.name}" Keras ImageNet Classifier'

    def predict(self, image_file_path: str):
        ...
