from src.models.module_img import ImgModule
from src.models.module_tab import TabModule
import open_clip


def get_model(config, device):
    arch = config['MODEL']['architecture']

    if arch in ["TSViT", "ConvLSTM", "ResNet", "ViT", "ViTFacto","MSClipFacto","L1C2L2AAdapterModel"]:
        return ImgModule(config)

    elif arch in [
        "TabTSViT",
        "TabConvLSTM",
        "MixResNet",
        "EnvResNet",
        "EnvViTFactorizeModel",
        "TabResNetConvLSTM",
        "TabViTFactorizeModel",
        "MultiViTFactorizeModel",
    ]:
        return TabModule(config).to(device)

    else:
        raise NameError(
            f"Model architecture {config['MODEL']['architecture']} not found, choose from: 'TSViT' or 'TabTSViT'"
        )
