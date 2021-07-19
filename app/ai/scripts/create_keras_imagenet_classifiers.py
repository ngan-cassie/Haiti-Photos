from ..models import KerasImageNetClassifier


def run():
    for model_class in KerasImageNetClassifier.MODELS_AND_INPUT_PREPROCESSORS:
        print(KerasImageNetClassifier.objects
              .update_or_create(name=model_class.__name__)[0])
