import torchvision
import torchvision.transforms as T

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

def cifar_linear() -> T.Compose:
    transform = T.Compose([
                T.RandomResizedCrop(size=32),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
            ])
    return transform
def cifar_test() -> T.Compose:
    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std'])
        ])
    
    return transform