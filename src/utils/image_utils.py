import torchvision.transforms as transforms
from PIL import Image, ImageFilter,ImageDraw, ImageFont
import random
import numpy as np
import cv2
from typing import Any

# 自定义转换：锐化
class Sharpen(transforms.Lambda):
    def __init__(self):
        super().__init__(self._apply_sharpen)

    def _apply_sharpen(self, img):
        return img.filter(ImageFilter.SHARPEN)


# 自定义转换：降噪（使用 OpenCV）
class RandomDenoise(transforms.Lambda):
    def __init__(self, h_range=(10, 30)):
        super().__init__(self._apply_denoise)
        self.h_range = h_range

    def _apply_denoise(self, img):
        img_np = np.array(img)
        # 从范围内随机选择降噪强度
        h = np.random.uniform(self.h_range[0], self.h_range[1])
        denoised_img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h, h, 7, 21)
        return Image.fromarray(denoised_img_np)


# 自定义转换：水印
class RandomWatermark:
    def __init__(self, text_list: list[str], font_size=20, opacity_range=(50, 150)):
        self.text_list: list[str] = text_list
        self.font_size = font_size
        self.opacity_range = opacity_range
        self.font = ImageFont.load_default()  # 使用默认字体，可替换为自定义字体

    def __call__(self, img):
        width, height = img.size
        watermark_img = Image.new("RGBA", img.size, (0, 0, 0, 0))

        # 随机水印大小
        scale_factor = random.uniform(0.05, 0.2)
        text_size = int(self.font_size * scale_factor)

        # 随机透明度
        opacity = random.randint(*self.opacity_range)

        # 随机水印内容
        text = self.text_list[random.randint(0, len(self.text_list) - 1)]

        # 创建水印
        watermark_draw = ImageDraw.Draw(watermark_img)
        bbox = watermark_draw.textbbox((0, 0), text, font=self.font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_img = Image.new("RGBA", (text_width, text_height), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), text, fill=(255, 255, 255, opacity), font=self.font)

        # 随机位置
        x = random.randint(0, max(0, width - text_width))
        y = random.randint(0, max(0, height - text_height))

        watermark_img.paste(text_img, (x, y), text_img)
        img = Image.alpha_composite(img.convert("RGBA"), watermark_img).convert("RGB")
        return img


class RemoveSolidEdgeRectangle:
    def __init__(self, tolerance: int = 0):
        self.tolerance = tolerance

    def __call__(self, img) -> Any:
        img = img.convert("RGB")
        data = np.array(img)
        height, width, _ = data.shape

        top, bottom, left, right = 0, height - 1, 0, width - 1

        if height > width:
            # 只检查上下边缘
            while top < height:
                if np.all(data[top, :, :] == data[top, 0, :]):  # 检查整行是否一致
                    top += 1
                else:
                    break

            while bottom >= 0:
                if np.all(data[bottom, :, :] == data[bottom, 0, :]):  # 检查整行是否一致
                    bottom -= 1
                else:
                    break

            # 确保裁剪后的高度 >= 宽度
            if (bottom - top + 1) < width:
                center = (top + bottom) // 2
                top = max(0, center - width // 2)
                bottom = min(height - 1, center + width // 2)

        else:
            # 只检查左右边缘
            while left < width:
                if np.all(data[:, left, :] == data[0, left, :]):  # 检查整列是否一致
                    left += 1
                else:
                    break

            while right >= 0:
                if np.all(data[:, right, :] == data[0, right, :]):  # 检查整列是否一致
                    right -= 1
                else:
                    break

            # 确保裁剪后的宽度 >= 高度
            if (right - left + 1) < height:
                center = (left + right) // 2
                left = max(0, center - height // 2)
                right = min(width - 1, center + height // 2)

        new_data = data[top : bottom + 1, left : right + 1, :]
        new_image = Image.fromarray(new_data, "RGB")
        return new_image
