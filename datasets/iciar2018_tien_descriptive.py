"""
Tien's ICIAR2018 dataset with DESCRIPTIVE class names.

Key features:
- Class registered as "ICIAR2018Descriptive"
- Uses split_fewshot_ga pickle directory (same as Tien)
- Uses split_iciar2018.json for full dataset splits
- Converts folder names to descriptive class names

To use: set DATASET.NAME to "ICIAR2018Descriptive" in the dataset config.
"""
import os
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

# Mapping from folder names to descriptive class names
NEW_CNAMES = {
    "Normal": "Normal breast tissue",
    "Benign": "Benign breast lesion",
    "InSitu": "In situ carcinoma",
    "Invasive": "Invasive carcinoma"
}


def _convert_classnames(data_list):
    """Convert folder-based class names to descriptive names."""
    converted = []
    for item in data_list:
        new_classname = NEW_CNAMES.get(item.classname, item.classname)
        converted.append(Datum(
            impath=item.impath,
            label=item.label,
            classname=new_classname
        ))
    return converted


@DATASET_REGISTRY.register()
class ICIAR2018Descriptive(DatasetBase):

    dataset_dir = "BACH/ICIAR2018_BACH_Challenge"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        # Note: Tien's pickle paths use "Photos" folder
        self.image_dir = os.path.join(self.dataset_dir, "Photos")
        # JSON and pickle files are in BACH/ not BACH/ICIAR2018_BACH_Challenge/
        bach_root = os.path.join(root, "BACH")
        self.split_path = os.path.join(bach_root, "split_breast_cancer.json")
        self.split_fewshot_dir = os.path.join(bach_root, "split_fewshot_ga")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            raise FileNotFoundError(
                f"Required split file not found: {self.split_path}\n"
                f"Please download the JSON split file for ICIAR2018."
            )

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # Convert class names to descriptive format
        train = _convert_classnames(train)
        val = _convert_classnames(val)
        test = _convert_classnames(test)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            if not os.path.isdir(category_dir):
                continue
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train : n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test
