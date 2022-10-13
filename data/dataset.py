
from PIL import Image
from torchvision import datasets
import torchvision.transforms as T

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class Cifar10Dataset(datasets.CIFAR10):
    def __init__(self,
        root: str,
        train: bool,
        transform: T.Compose,
        **kwargs):
        
        super(Cifar10Dataset, self).__init__(root, train=train, transform=transform, **kwargs)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]  # img:(H, W, C)=(32, 32, 3)
    
        img = Image.fromarray(img)

        img1 = self.transform(img)
        img2 = self.transform(img)

        return (index, ) + (img1, img2)

    
class OURDataset(datasets.CIFAR10):
    def __init__(self,
        root: str,
        train: bool,
        transform: T.Compose,
        **kwargs):
        
        super(Cifar10Dataset, self).__init__(root, train=train, transform=transform, **kwargs)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]  # img:(H, W, C)=(32, 32, 3)
    
        img = Image.fromarray(img)

        img1 = self.transform_inv(img)
        img2 = self.transform_inv(img)

        img1 = self.transform_eqv(img1)
        img2 = self.transform_eqv(img2)
        
        return (index, ) + (img1, img2)

    def transform_inv(self, image):
        color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        trans_list = [
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomGrayscale(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=imagenet_normalize['mean'], std=imagenet_normalize['std'])
            
        ]
        transform = T.Compose(trans_list)

        return transform(image)

    def transform_eqv(self, image):
        trans_list=[
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0)
        ]

        transform = T.Compose(trans_list)

        return transform(image)