import timm

def uni():

    # login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    # pretrained=True needed to load UNI weights (and download weights for the first time)
    # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, dynamic_img_size=True)
    # transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    # model.eval()
    return model
