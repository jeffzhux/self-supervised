import torchvision
import torchvision.transforms as T
from data.transforms import GaussianBlur, Jigsaw, RandomRotate, RandomSolarization

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

class BaseTransform():
    """Implementation of a collate function for images.
    This is an implementation of the BaseCollateFunction with a concrete
    set of transforms.
    The set of transforms is inspired by the SimCLR paper as it has shown
    to produce powerful embeddings. 
    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Sigma of gaussian blur is kernel_size * input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random (+90 degree) rotation is applied.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """
    def __init__(
        self,
        input_size: int = 64,
        cj_prob: float = 0.8,
        cj_bright: float = 0.7,
        cj_contrast: float = 0.7,
        cj_sat: float = 0.7,
        cj_hue: float = 0.2,
        min_scale: float = 0.15,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: float = 0.1,
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        normalize: dict = imagenet_normalize):

        color_jitter = T.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue
        )
        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            RandomRotate(prob=rr_prob),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(
                kernel_size=kernel_size * input_size,
                prob=gaussian_blur),
            T.ToTensor()
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize['mean'], std=normalize['std'])]
           
        self.transform = T.Compose(transform)
    def __call__(self, x) -> T.Compose:
        return self.transform(x)

class SimCLRTransform(BaseTransform):
    """
    Examples:
        >>> # SimCLR for ImageNet
        >>> collate_fn = SimCLRCollateFunction()
        >>> 
        >>> # SimCLR for CIFAR-10
        >>> collate_fn = SimCLRCollateFunction(
        >>>     input_size=32,
        >>>     gaussian_blur=0.,
        >>> )
    """
    def __init__(self,
                 input_size: int = 224,
                 cj_prob: float = 0.8,
                 cj_strength: float = 0.5,
                 min_scale: float = 0.08,
                 random_gray_scale: float = 0.2,
                 gaussian_blur: float = 0.5,
                 kernel_size: float = 0.1,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 normalize: dict = imagenet_normalize):

        super(SimCLRTransform, self).__init__(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_bright=cj_strength * 0.8,
            cj_contrast=cj_strength * 0.8,
            cj_sat=cj_strength * 0.8,
            cj_hue=cj_strength * 0.2,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            normalize=normalize,
        )
