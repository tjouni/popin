import os
import boto3
import torch
import json
import re
import pytorch_lightning
import torch.nn as nn
import torchvision.transforms as T
from lightning import Insta
from io import BytesIO

import base64
import PIL


S3 = boto3.client("s3")


def inverse_transform(x):
    return -torch.log(x) / 7


def handler(event, context):
    print(json.dumps(event))
    base64_data = re.sub(
        "^data:image/.+;base64,", "", json.loads(event["body"])["body"]
    )
    image_string = base64.b64decode(base64_data)
    filename = "/tmp/kuva.jpg"
    with open(filename, "wb+") as image_file:
        image_file.write(image_string)
    img = PIL.Image.open(filename)
    model = Predictor()
    result = model.forward(img)
    result = float(result[0][0])
    return {"statusCode": 200, "body": result}


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        filename = f"/tmp/model.ckpt"
        S3.download_file(
            "popin-data-bucket", "pretrained_without_tt.ckpt", Filename=filename
        )
        model = torch.load(filename, map_location=torch.device("cpu"))
        self.net = Insta.load_from_checkpoint(filename)
        if self.net.hparams.pretrained:
            self.preprocess = T.Compose(
                [
                    T.ToTensor(),
                    T.Resize(400),
                    T.CenterCrop(400),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.preprocess = T.Compose(
                [T.ToTensor(), T.Resize(400), T.CenterCrop(400)]
            )

    def forward(self, pil_image):
        img = self.preprocess(pil_image)
        batch = img.unsqueeze(0)
        with torch.inference_mode():
            if self.net.hparams.transform_target:
                result = inverse_transform(self.net(batch))
            else:
                result = self.net(batch)
        return result
