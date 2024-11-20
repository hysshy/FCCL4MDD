import os
from torch.utils.data import Dataset
from PIL import Image

class SynLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for label in os.listdir(root_dir):
            for filename in os.listdir(os.path.join(root_dir, label)):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.JPG')):
                    img_path = os.path.join(root_dir, label, filename)
                    self.samples.append((img_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, labels


