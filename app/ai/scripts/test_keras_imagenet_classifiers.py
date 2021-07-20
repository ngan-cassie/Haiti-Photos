from ..models import KerasImageNetClassifier

from tqdm import tqdm


def run(image_file_path: str):
    for model in tqdm(KerasImageNetClassifier.objects.all(),
                      total=KerasImageNetClassifier.objects.count()):
        print(model)
        print(model.predict(image_file_path=image_file_path, n_labels=5))
