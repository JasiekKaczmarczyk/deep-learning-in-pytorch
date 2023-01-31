import torch
import torchvision.transforms as T
import numpy as np

class Transforms:
    def __init__(self):
        self.i2t = T.Compose(
            [
                T.ToTensor(),
                T.Resize((256, 256)),
                T.Lambda(lambda t: 2 * t -1)
            ]
        )
        self.t2i = T.Compose(
            [
                T.Lambda(lambda t: (t + 1) / 2),
                T.Lambda(lambda t: t.permute(1, 2, 0)),
                T.Lambda(lambda t: t * 255),
                T.Lambda(lambda t: t.numpy().astype(np.uint8)),
                T.ToPILImage()
            ]
        )

    def img2torch(self, x: np.array):
        return self.i2t(x)

    def torch2img(self, x: torch.Tensor):
        return self.t2i(x)
