from ..models import KerasImageNetClassifier

from tqdm import tqdm


def run(*image_file_paths: str):
    for image_file_path in tqdm(image_file_paths):
        for model in tqdm(KerasImageNetClassifier.objects.all(),
                          total=KerasImageNetClassifier.objects.count()):
            print(model)
            print(model.predict(image_file_path=image_file_path, n_labels=5))
