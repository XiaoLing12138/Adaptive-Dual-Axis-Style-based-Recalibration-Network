import timm


def MLP_mixer(num_classes=3):
    return timm.create_model('mixer_s16_224', num_classes=num_classes)


if __name__ == '__main__':
    net = MLP_mixer()
    print(net)
