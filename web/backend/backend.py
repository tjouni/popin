# TODO: deploy as lambda, remove unnecessary code

import os
import boto3
import torch
import pytorch_lightning
import torch.nn as nn
import torchvision.transforms as T
from lightning import Insta
from io import BytesIO


import base64
import PIL


S3 = boto3.client("s3")


def get_fake_event():
    with open("2940801053738400977.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return {"body": encoded_string}


def inverse_transform(x):
    return 0.005 * (200 - 13 * x) / x / 6.8214285714 - 0.060775


def handler(event, context):
    image_string = base64.b64decode(event["body"])
    filename = "kuva.jpg"
    with open("kuva.jpg", "wb") as image_file:
        image_file.write(image_string)
    img = PIL.Image.open(filename)
    img.show()
    model = Predictor()
    result = model.forward(img)
    print(float(inverse_transform(result[0][0])))


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        filename = f"/tmp/{os.path.basename('model.ckpt')}"
        S3.download_file(
            "jounin-testi-ampari", "preliminary_weights.ckpt", Filename=filename
        )
        model = torch.load(filename, map_location=torch.device("cpu"))
        self.net = Insta.load_from_checkpoint(filename)
        self.preprocess = T.Compose([T.ToTensor(), T.Resize(400), T.CenterCrop(400)])

    def forward(self, pil_image):
        img = self.preprocess(pil_image)
        batch = img.unsqueeze(0)
        with torch.inference_mode():
            if self.net.hparams.transform_target:
                result = inverse_transform(self.net(batch))
            else:
                result = self.net(batch)
        return result


event = get_fake_event()
handler(event, None)
