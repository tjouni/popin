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
    return 0.005 * (200 - 13 * x) / x / 6.8214285714 - 0.060775


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
    result = min((float(result[0][0]) / 1.5), 1)
    return {"statusCode": 200, "body": result}


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        filename = f"/tmp/{os.path.basename('model.ckpt')}"
        S3.download_file(
            "popin-data-bucket", "preliminary_weights.ckpt", Filename=filename
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
