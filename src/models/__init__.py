from src.models.module_img import ImgModule
from src.models.module_tab import TabModule
from src.models.msclip_factorize_model import MSClipFactorizeModel
import open_clip

def get_model(config, device):
    arch = config['MODEL']['architecture']

    if arch == "MSClipFactorize":
        model_cfg = config["MODEL"]
        # pass relevant args
        net = MSClipFactorizeModel(
            model_name=model_cfg.get("msclip_model_name", "Llama3-MS-CLIP-Base"),
            ckpt_path=model_cfg.get("msclip_ckpt", None),
            channels=model_cfg.get("input_dim", 14),
            num_classes=model_cfg.get("num_classes", 2),
            out_H=model_cfg.get("out_H", 25),
            out_W=model_cfg.get("out_W", 25),
            temp_enc_type=model_cfg.get("temp_enc_type", "attention"),
            temp_depth=model_cfg.get("temp_depth", 2),
            use_conv_decoder=model_cfg.get("use_conv_decoder", False),
            freeze_msclip=model_cfg.get("freeze_msclip", True),
        )
        return net.to(device)
    
    if arch in ["TSViT", "ConvLSTM", "ResNet", "ViT", "ViTFacto"]:
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
