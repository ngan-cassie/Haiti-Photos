from ..models import KerasImageNetClassifier

from tqdm import tqdm


def run():
    for model in tqdm(KerasImageNetClassifier.objects.all(),
                      total=KerasImageNetClassifier.objects.count()):
        print(model)
