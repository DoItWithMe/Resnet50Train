import torch.nn.functional as nnF
import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger as log
from .orthonet_mod import orthonet_mod_50

# from .attention import modify_resnet_with_se
import timm


class SiameseNetworkL2Net1(nn.Module):
    def __init__(self, resetnet50_weight_path):
        super(SiameseNetworkL2Net1, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        self.resnet = models.resnet50(weights=None)
        pretrained_weight = torch.load(resetnet50_weight_path, weights_only=True)
        self.resnet.load_state_dict(pretrained_weight["state_dict"], strict=False)

        # 移除 ResNet50 的全连接层 和 池化层
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])

        self.fc1 = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self._init_params()

    def _init_params(self):
        # Initialize the weights for the fully connected layers
        for m in self.fc1:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nnF.normalize(x)
        return x

    def check(self, output):
        if torch.isnan(output).any():
            log.warning(f"output have nan....")

        if torch.isinf(output).any() > 0:
            log.warning(f"output have inf....")

    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)

        self.check(output1)
        self.check(output2)
        self.check(output3)

        return output1, output2, output3

    def extractor_features(self, x):
        results_map = dict()

        # lay4_f = self.feature_extractor(x)
        # results_map["resnet50_lay4_output"] = flatten_normalize(lay4_f)

        fc_f = self.forward_one(x)
        results_map["resnet50_fc_output"] = fc_f
        return results_map


class SiameseNetworkL2Net1WithSE(nn.Module):
    def __init__(self, resetnet50_weight_path):
        super(SiameseNetworkL2Net1WithSE, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        # 直接使用 timm 中的 支持 SE 模块的 seresnet50
        self.resnet = timm.create_model("resnetv2_50x1_bit", pretrained=False)
        pretrained_weight = torch.load(
            resetnet50_weight_path, weights_only=True, map_location="cpu"
        )
        self.resnet.load_state_dict(pretrained_weight, strict=False)

        # log.info(f"{self.resnet}")
        # 移除 ResNet50 的全连接层 和 池化层
        self.feature_extractor = nn.Sequential(
            *list(self.resnet.stem.children()),
            *list(self.resnet.stages.children()),
            self.resnet.norm,
            *list(self.resnet.head.children())[:-2],
        )
        # self.feature_extractor = nn.Sequential(*list(self.resnet.children()))

        log.info(f"{self.feature_extractor}")

        # self.fc1 = nn.Sequential(
        #     nn.Linear(2048 * 7 * 7, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     # nn.BatchNorm1d(256),
        #     # nn.ReLU(),
        #     # nn.Linear(256, 128),
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
        )

        self._init_params()

    def _init_params(self):
        # Initialize the weights for the fully connected layers
        for m in self.fc1:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        x = self.feature_extractor(x)
        # log.info(f"x: {x.shape}")
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nnF.normalize(x)
        return x

    def check(self, output):
        if torch.isnan(output).any():
            log.warning(f"output have nan....")

        if torch.isinf(output).any() > 0:
            log.warning(f"output have inf....")

    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)

        self.check(output1)
        self.check(output2)
        self.check(output3)

        return output1, output2, output3

    def extractor_features(self, x):
        results_map = dict()

        # lay4_f = self.feature_extractor(x)
        # results_map["resnet50_lay4_output"] = flatten_normalize(lay4_f)

        fc_f = self.forward_one(x)
        results_map["resnet50_fc_output"] = fc_f
        return results_map


class SiameseNetworkL2Net1WithOrthoNet(nn.Module):
    def __init__(self, resetnet50_weight_path):
        super(SiameseNetworkL2Net1WithOrthoNet, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        # 直接使用 timm 中的 支持 SE 模块的 seresnet50
        self.resnet = timm.create_model("resnetv2_50x1_bit", pretrained=False)
        self.resnet = orthonet_mod_50(n_classes=1000)
        pretrained_weight = torch.load(
            resetnet50_weight_path, weights_only=True, map_location="cpu"
        )
        self.resnet.load_state_dict(pretrained_weight, strict=False)

        # log.info(f"{self.resnet}")
        # 移除 ResNet50 的全连接层 和 池化层
        # self.feature_extractor = nn.Sequential(
        #     *list(self.resnet.stem.children()),
        #     *list(self.resnet.stages.children()),
        #     self.resnet.norm,
        #     *list(self.resnet.head.children())[:-2]
        # )
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])

        log.info(self.feature_extractor)
        self.fc1 = nn.Sequential(
            nn.Linear(2048 * 8 * 8, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
        )

        # self.fc1 = nn.Sequential(
        #     nn.Linear(2048, 512,bias=False),
        #     nn.BatchNorm1d(512),
        #     # nn.ReLU(),
        #     # nn.Linear(1024, 512),
        #     # nn.BatchNorm1d(512),
        # )

        self._init_params()

    def _init_params(self):
        # Initialize the weights for the fully connected layers
        for m in self.fc1:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        x = self.feature_extractor(x)
        # log.info(f"x: {x.shape}")
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nnF.normalize(x)
        return x

    def check(self, output):
        if torch.isnan(output).any():
            log.warning(f"output have nan....")

        if torch.isinf(output).any() > 0:
            log.warning(f"output have inf....")

    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)

        self.check(output1)
        self.check(output2)
        self.check(output3)

        return output1, output2, output3

    def extractor_features(self, x):
        results_map = dict()

        # lay4_f = self.feature_extractor(x)
        # results_map["resnet50_lay4_output"] = flatten_normalize(lay4_f)

        fc_f = self.forward_one(x)
        results_map["resnet50_fc_output"] = fc_f
        return results_map


class SiameseNetworkL2Net2(nn.Module):
    def __init__(self):
        super(SiameseNetworkL2Net2, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(self.base_model.children())[:-2])

        self.fc1 = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
        )

        self._init_params()

    def _init_params(self):
        # Initialize the weights for the fully connected layers
        for m in self.fc1:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nnF.normalize(x)
        return x

    def check(self, output):
        if torch.isnan(output).any():
            log.warning(f"output have nan....")

        if torch.isinf(output).any() > 0:
            log.warning(f"output have inf....")

    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)

        self.check(output1)
        self.check(output2)
        self.check(output3)

        return output1, output2, output3
