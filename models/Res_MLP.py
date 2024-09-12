import timm


def Res_MLP(num_classes=3):
    return timm.create_model('resmlp_12_distilled_224', num_classes=num_classes)
