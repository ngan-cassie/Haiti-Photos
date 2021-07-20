import json
import numpy
from pathlib import Path
from PIL import Image, ImageOps
from typing import Dict

from h1st.django.model.api import H1stModel

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
from tensorflow.keras.applications.xception import (
    Xception,
    preprocess_input as xception_preprocess_input
)


IMAGENET_LABELS = json.load(open(Path(__file__).parent.parent
                                 / 'ImageNet-Labels.json', 'r'))


class KerasImageNetClassifier(H1stModel):
    MODEL_CLASSES_AND_IMAGE_SIZES_AND_INPUT_PREPROCESSORS = {
        'DenseNet121': (DenseNet121,
                        (224, 224),
                        densenet_preprocess_input),
        'DenseNet169': (DenseNet169,
                        (224, 224),
                        densenet_preprocess_input),
        'DenseNet201': (DenseNet201,
                        (224, 224),
                        densenet_preprocess_input),

        'EfficientNetB0': (EfficientNetB0,
                           (224, 224),
                           efficientnet_preprocess_input),
        'EfficientNetB1': (EfficientNetB1,
                           (240, 240),
                           efficientnet_preprocess_input),
        'EfficientNetB2': (EfficientNetB2,
                           (260, 260),
                           efficientnet_preprocess_input),
        'EfficientNetB3': (EfficientNetB3,
                           (300, 300),
                           efficientnet_preprocess_input),
        'EfficientNetB4': (EfficientNetB4,
                           (380, 380),
                           efficientnet_preprocess_input),
        'EfficientNetB5': (EfficientNetB5,
                           (456, 456),
                           efficientnet_preprocess_input),
        'EfficientNetB6': (EfficientNetB6,
                           (528, 528),
                           efficientnet_preprocess_input),
        'EfficientNetB7': (EfficientNetB7,
                           (600, 600),
                           efficientnet_preprocess_input),

        'InceptionResNetV2': (InceptionResNetV2,
                              (299, 299),
                              inception_resnet_v2_preprocess_input),
        'InceptionV3': (InceptionV3,
                        (299, 299),
                        inception_v3_preprocess_input),

        'MobileNet': (MobileNet,
                      (224, 224),
                      mobilenet_preprocess_input),
        'MobileNetV2': (MobileNetV2,
                        (224, 224),
                        mobilenet_v2_preprocess_input),
        'MobileNetV3Large': (MobileNetV3Large,
                             (224, 224),
                             mobilenet_v3_preprocess_input),
        'MobileNetV3Small': (MobileNetV3Small,
                             (224, 224),
                             mobilenet_v3_preprocess_input),

        'NASNetLarge': (NASNetLarge,
                        (331, 331),
                        nasnet_preprocess_input),
        'NASNetMobile': (NASNetMobile,
                         (224, 224),
                         nasnet_preprocess_input),

        'ResNet50': (ResNet50,
                     (224, 224),
                     resnet_preprocess_input),
        'ResNet101': (ResNet101,
                      (224, 224),
                      resnet_preprocess_input),
        'ResNet152': (ResNet152,
                      (224, 224),
                      resnet_preprocess_input),
        'ResNet50V2': (ResNet50V2,
                       (224, 224),
                       resnet_v2_preprocess_input),
        'ResNet101V2': (ResNet101V2,
                        (224, 224),
                        resnet_v2_preprocess_input),
        'ResNet152V2': (ResNet152V2,
                        (224, 224),
                        resnet_v2_preprocess_input),

        'VGG16': (VGG16,
                  (224, 224),
                  vgg16_preprocess_input),
        'VGG19': (VGG19,
                  (224, 224),
                  vgg19_preprocess_input),

        'Xception': (Xception,
                     (299, 299),
                     xception_preprocess_input)
    }

    def save(self, *args, **kwargs):
        assert self.name in \
            self.MODEL_CLASSES_AND_IMAGE_SIZES_AND_INPUT_PREPROCESSORS, \
            f'*** {self.name} INVALID ***'
        super().save(*args, **kwargs)

    def load(self):
        assert self.name in \
            self.MODEL_CLASSES_AND_IMAGE_SIZES_AND_INPUT_PREPROCESSORS, \
            f'*** {self.name} INVALID ***'

        self.model_class, self.image_size, self.preprocessor = \
            self.MODEL_CLASSES_AND_IMAGE_SIZES_AND_INPUT_PREPROCESSORS[
                self.name]

        self.model_obj = self.model_class()

    def predict(self, image_file_path: str, n_labels=5) -> Dict[str, float]:
        self.load()

        # load image
        image = Image.open(fp=image_file_path, mode='r', formats=None)

        # scale image to size model expects
        scaled_image = ImageOps.fit(image=image,
                                    size=self.image_size,
                                    centering=(0.5, 0.5))

        # convert image to NumPy array
        scaled_image_array = numpy.asarray(scaled_image, dtype=int, order=None)

        # make a batch of 1 array
        img_batch_arr = numpy.expand_dims(scaled_image_array, axis=0)

        # preprocess
        prep_img_batch_arr = self.preprocessor(img_batch_arr)

        # predict
        predictions = self.model_obj.predict(x=prep_img_batch_arr)

        # pair labels & predictions
        labels_and_predictions = zip(IMAGENET_LABELS, predictions.flatten())

        # sort label-prediction pairs by decreasing probability
        sorted_labels_and_predictions = \
            sorted(labels_and_predictions,
                   key=lambda label, prediction: prediction,
                   reverse=True)

        # return JSON dict
        return dict(sorted_labels_and_predictions[:n_labels])
