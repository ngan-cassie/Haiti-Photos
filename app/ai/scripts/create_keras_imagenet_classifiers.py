from ..models import KerasImageNetClassifier


def run():
    for model_class_name in \
            KerasImageNetClassifier.MODEL_CLASSES_AND_IMAGE_SIZES_AND_INPUT_PREPROCESSORS:
        print(KerasImageNetClassifier.objects
              .update_or_create(name=model_class_name)[0])
