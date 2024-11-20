import numpy as np
from torchvision import datasets, transforms
import os

data_dir = os.path.join("data")


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class Derm7pt(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=(64, 64), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    test_trsf = [
            transforms.Resize(size=(64, 64), interpolation=3),
            transforms.ToTensor()
            ]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ),
    ]

    class_order = np.arange(20).tolist()
    
    def download_data(self, trainPath=None, testPath=None):
        train_dataset = datasets.ImageFolder(trainPath)
        train_data = []
        train_targets = []
        for i in range(len(train_dataset.imgs)):
            train_data.append(train_dataset.imgs[i][0])
            train_targets.append(train_dataset.imgs[i][1])
        self.train_data = np.array(train_data)
        self.train_targets = np.array(train_targets)

        test_dataset = datasets.ImageFolder(testPath)
        test_targets = []
        test_data = []
        for i in range(len(test_dataset.imgs)):
            test_data.append(test_dataset.imgs[i][0])
            test_targets.append(test_dataset.imgs[i][1])
        self.test_data = np.array(test_data)
        self.test_targets = np.array(test_targets)


class JSIEC(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=(64, 64), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    test_trsf = [
        transforms.Resize(size=(64, 64), interpolation=3),
        transforms.ToTensor()
    ]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ),
    ]

    class_order = np.arange(39).tolist()

    def download_data(self, trainPath=None, testPath=None):
        train_dataset = datasets.ImageFolder(trainPath)
        train_data = []
        train_targets = []
        for i in range(len(train_dataset.imgs)):
            train_data.append(train_dataset.imgs[i][0])
            train_targets.append(train_dataset.imgs[i][1])
        self.train_data = np.array(train_data)
        self.train_targets = np.array(train_targets)

        test_dataset = datasets.ImageFolder(testPath)
        test_targets = []
        test_data = []
        for i in range(len(test_dataset.imgs)):
            test_data.append(test_dataset.imgs[i][0])
            test_targets.append(test_dataset.imgs[i][1])
        self.test_data = np.array(test_data)
        self.test_targets = np.array(test_targets)