from h1st.django.model.api import H1stModel

from django.db.models.fields import CharField

from tensorflow.keras.applications.densenet import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    preprocess_input as densenet_preprocess_input
)
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    preprocess_input as efficientnet_preprocess_input
)
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input as inception_resnet_v2_preprocess_input
)
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input as inception_v3_preprocess_input
)
from tensorflow.keras.applications.mobilenet import (
    MobileNet,
    preprocess_input as mobilenet_preprocess_input
)
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_v2_preprocess_input
)
from tensorflow.keras.applications import (
    MobileNetV3Large,
    MobileNetV3Small
)
from tensorflow.keras.applications.mobilenet_v3 import (
    preprocess_input as mobilenet_v3_preprocess_input
)
from tensorflow.keras.applications.nasnet import (
    NASNetLarge,
    NASNetMobile,
    preprocess_input as nasnet_preprocess_input
)
from tensorflow.keras.applications.resnet import (
    ResNet50,
    ResNet101,
    ResNet152,
    preprocess_input as resnet_preprocess_input
)
from tensorflow.keras.applications.resnet_v2 import (
    ResNet50V2,
    ResNet101V2,
    ResNet152V2,
    preprocess_input as resnet_v2_preprocess_input
)
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as vgg16_preprocess_input
)
from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input as vgg19_preprocess_input
)
from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input as vgg19_preprocess_input
)
from tensorflow.keras.applications.xception import (
    Xception,
    preprocess_input as xception_preprocess_input
)


class KerasImageNetClassifier(H1stModel):
    MODELS_AND_INPUT_PREPROCESSORS = {
        DenseNet121: densenet_preprocess_input,
        DenseNet169: densenet_preprocess_input,
        DenseNet201: densenet_preprocess_input,

        EfficientNetB0: efficientnet_preprocess_input,
        EfficientNetB1: efficientnet_preprocess_input,
        EfficientNetB2: efficientnet_preprocess_input,
        EfficientNetB3: efficientnet_preprocess_input,
        EfficientNetB4: efficientnet_preprocess_input,
        EfficientNetB5: efficientnet_preprocess_input,
        EfficientNetB6: efficientnet_preprocess_input,
        EfficientNetB7: efficientnet_preprocess_input,

        InceptionResNetV2: inception_resnet_v2_preprocess_input,
        InceptionV3: inception_v3_preprocess_input,

        MobileNet: mobilenet_preprocess_input,
        MobileNetV2: mobilenet_v2_preprocess_input,
        MobileNetV3Large: mobilenet_v3_preprocess_input,
        MobileNetV3Small: mobilenet_v3_preprocess_input,

        NASNetLarge: nasnet_preprocess_input,
        NASNetMobile: nasnet_preprocess_input,

        ResNet50: resnet_preprocess_input,
        ResNet101: resnet_preprocess_input,
        ResNet152: resnet_preprocess_input,

        ResNet50V2: resnet_v2_preprocess_input,
        ResNet101V2: resnet_v2_preprocess_input,
        ResNet152V2: resnet_v2_preprocess_input,

        VGG16: vgg16_preprocess_input,
        VGG19: vgg19_preprocess_input,

        Xception: xception_preprocess_input
    }

    name = CharField(
            verbose_name='Keras ImageNet Classifier Name',
            help_text='Keras ImageNet Classifier Name',
            max_length=255,
            null=False,
            blank=False,
            choices=[
                ('DenseNet121', 'DenseNet121'),
                ('DenseNet169', 'DenseNet169'),
                ('DenseNet201', 'DenseNet201'),

                ('EfficientNetB0', 'EfficientNetB0'),
                ('EfficientNetB1', 'EfficientNetB1'),
                ('EfficientNetB2', 'EfficientNetB2'),
                ('EfficientNetB3', 'EfficientNetB3'),
                ('EfficientNetB4', 'EfficientNetB4'),
                ('EfficientNetB5', 'EfficientNetB5'),
                ('EfficientNetB6', 'EfficientNetB6'),
                ('EfficientNetB7', 'EfficientNetB7'),

                ('InceptionResNetV2', 'InceptionResNetV2'),
                ('InceptionV3', 'InceptionV3'),

                ('MobileNet', 'MobileNet'),
                ('MobileNetV2', 'MobileNetV2'),
                ('MobileNetV3Large', 'MobileNetV3Large'),
                ('MobileNetV3Small', 'MobileNetV3Small'),

                ('NASNetLarge', 'NASNetLarge'),
                ('NASNetMobile', 'NASNetMobile'),

                ('ResNet50', 'ResNet50'),
                ('ResNet101', 'ResNet101'),
                ('ResNet152', 'ResNet152'),

                ('ResNet50V2', 'ResNet50V2'),
                ('ResNet101V2', 'ResNet101V2'),
                ('ResNet152V2', 'ResNet152V2'),

                ('VGG16', 'VGG16'),
                ('VGG19', 'VGG19'),

                ('Xception', 'Xception')
            ],
            db_index=True,
            default=None,
            editable=True,
            unique=True)

    class Meta:
        verbose_name = 'Keras ImageNet Classifier'
        verbose_name_plural = 'Keras ImageNet Classifiers'

    def __str__(self) -> str:
        return f'"{self.name}" Keras ImageNet Classifier'

    def predict(self, image_file_path: str):
        ...
