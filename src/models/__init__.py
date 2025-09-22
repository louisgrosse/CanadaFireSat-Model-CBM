from src.models.module_img import ImgModule
from src.models.module_tab import TabModule
from src.models.msclip_factorize_model import MSClipFactorizeModel
import open_clip

def get_model(config, device):
    arch = config['MODEL']['architecture']

    if arch in ["TSViT", "ConvLSTM", "ResNet", "ViT", "ViTFacto","MSClipFacto"]:
        return ImgModule(config).to(device)

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
