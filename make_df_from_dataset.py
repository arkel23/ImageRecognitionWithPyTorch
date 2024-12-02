# https://github.com/pytorch/vision/tree/main/torchvision/datasets
# https://github.com/lvyilin/pytorch-fgvc-dataset
import os
import argparse
import pathlib
from typing import Any, Callable, Optional, Union, Tuple, Sequence

import pandas as pd

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg, \
    download_file_from_google_drive


class CUB(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    # file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    # filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CUB, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(self.url, download_root=self.root, md5=self.tgz_md5)
        # download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        #with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
        #    tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_dir = sample.filepath
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        return target, img_dir


class Pets(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:
                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.
            If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [f'{image_id}.jpg' for image_id in image_ids]
        # self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_dir = self._images[idx]
        target = self._labels[idx]
        return target, image_dir

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)


def make_df_from_dataset(args):

    if args.dataset_name == 'cub':
        base_folder = 'CUB_200_2011'
        dataset_train = CUB(root=args.dataset_root_path, train=True, download=True)
        dataset_test = CUB(root=args.dataset_root_path, train=False, download=True)
        print(len(dataset_train), dataset_train[0], len(dataset_train.class_names))
    elif args.dataset_name == 'pets':
        base_folder = 'oxford-iiit-pet'
        dataset_train = Pets(root=args.dataset_root_path, split='trainval', download=True)
        dataset_test = Pets(root=args.dataset_root_path, split='test', download=True)
    else:
        raise NotImplementedError

    print('Test set:\n', len(dataset_test), dataset_test[0])

    for (split, dataset) in zip((args.save_name_train, args.save_name_test),
                                (dataset_train, dataset_test)):
        dic_target_img_dir = {}
        for i, (target, img_dir) in enumerate(dataset):
            dic_target_img_dir[i] = {'class_id': target, 'dir': img_dir}

        df = pd.DataFrame.from_dict(dic_target_img_dir, orient='index')
        print(f'Split: {split}\n', df.head(), len(df))

        fp = os.path.join(args.dataset_root_path, base_folder, f'{split}.csv')
        df.to_csv(fp, header=True, index=False)

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required=True,
                        help='path to dataset root')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--save_name_train', type=str, default='train_val')
    parser.add_argument('--save_name_test', type=str, default='test')
    args = parser.parse_args()
    make_df_from_dataset(args)


if __name__ == '__main__':
    main()
