import torchvision.transforms as transforms
from augly.image import (
    EncodingQuality,
    OneOf,
    RandomBlur,
    RandomEmojiOverlay,
    RandomPixelization,
    RandomRotation,
    ShufflePixels,
)
from augly.image.transforms import BaseTransform
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
from augly.image.functional import overlay_emoji, overlay_image, overlay_text

# class RandomOverlayText(BaseTransform):
#     def __init__(
#         self,
#         opacity: float = 1.0,
#         p: float = 1.0,
#     ):
#         super().__init__(p)
#         self.opacity = opacity

#         with open(Path(FONTS_DIR) / FONT_LIST_PATH) as f:
#             font_list = [s.strip() for s in f.readlines()]
#             blacklist = [
#                 'TypeMyMusic',
#                 'PainttheSky-Regular',
#             ]
#             self.font_list = [
#                 f for f in font_list
#                 if all(_ not in f for _ in blacklist)
#             ]

#         self.font_lens = []
#         for ff in self.font_list:
#             font_file = Path(MODULE_BASE_DIR) / ff.replace('.ttf', '.pkl')
#             with open(font_file, 'rb') as f:
#                 self.font_lens.append(len(pickle.load(f)))


class RandomEdgeEnhance(BaseTransform):
    def __init__(
        self,
        mode=ImageFilter.EDGE_ENHANCE,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.mode = mode

    def apply_transform(self, image: Image.Image, *args) -> Image.Image:
        return image.filter(self.mode)


class RandomOverlayImageAndResizedCrop(BaseTransform):
    def __init__(
        self,
        img_paths: List[Path],
        opacity_lower: float = 0.5,
        size_lower: float = 0.4,
        size_upper: float = 0.6,
        input_size: int = 224,
        moderate_scale_lower: float = 0.7,
        hard_scale_lower: float = 0.15,
        overlay_p: float = 0.05,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.img_paths = img_paths
        self.opacity_lower = opacity_lower
        self.size_lower = size_lower
        self.size_upper = size_upper
        self.input_size = input_size
        self.moderate_scale_lower = moderate_scale_lower
        self.hard_scale_lower = hard_scale_lower
        self.overlay_p = overlay_p

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        # print(f"apply_transform called with args: {locals()}")
        if random.uniform(0.0, 1.0) < self.overlay_p:
            if random.uniform(0.0, 1.0) > 0.5:
                background = Image.open(random.choice(self.img_paths))
                overlay = image
            else:
                background = image
                overlay = Image.open(random.choice(self.img_paths))

            overlay_size = random.uniform(self.size_lower, self.size_upper)
            image = overlay_image(
                background,
                overlay=overlay,
                opacity=random.uniform(self.opacity_lower, 1.0),
                overlay_size=overlay_size,
                x_pos=random.uniform(0.0, 1.0 - overlay_size),
                y_pos=random.uniform(0.0, 1.0 - overlay_size),
                metadata=metadata,
            )
            return transforms.RandomResizedCrop(
                self.input_size, scale=(self.moderate_scale_lower, 1.0)
            )(image)
        else:
            return transforms.RandomResizedCrop(
                self.input_size, scale=(self.hard_scale_lower, 1.0)
            )(image)


class ShuffledAug:

    def __init__(self, aug_list):
        self.aug_list = aug_list

    def __call__(self, x):
        # without replacement
        shuffled_aug_list = random.sample(self.aug_list, len(self.aug_list))
        for op in shuffled_aug_list:
            x = op(x)
        return x


def convert2rgb(x):
    return x.convert("RGB")


# aug_moderate = [
#     transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ]

aug_list = [
    transforms.ColorJitter(0.7, 0.7, 0.7, 0.2),
    RandomPixelization(p=0.25),
    ShufflePixels(factor=0.1, p=0.25),
    OneOf([EncodingQuality(quality=q) for q in [10, 20, 30, 50]], p=0.25),
    transforms.RandomGrayscale(p=0.25),
    RandomBlur(p=0.25),
    transforms.RandomPerspective(p=0.25),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    # RandomOverlayText(p=0.25),
    RandomEmojiOverlay(p=0.25),
    OneOf(
        [
            RandomEdgeEnhance(mode=ImageFilter.EDGE_ENHANCE),
            RandomEdgeEnhance(mode=ImageFilter.EDGE_ENHANCE_MORE),  # type:ignore
        ],  # type:ignore
        p=0.25,
    ),
]

# aug_hard = [
#     RandomRotation(p=0.25),
#     RandomOverlayImageAndResizedCrop(
#         train_paths,
#         opacity_lower=0.6,
#         size_lower=0.4,
#         size_upper=0.6,
#         input_size=224,
#         moderate_scale_lower=0.7,
#         hard_scale_lower=0.15,
#         overlay_p=0.05,
#         p=1.0,
#     ),
#     ShuffledAug(aug_list),
#     convert2rgb,
#     transforms.ToTensor(),
#     transforms.RandomErasing(value="random", p=0.25),  # type:ignore
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ]
