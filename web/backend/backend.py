# TODO: deploy as lambda, remove unnecessary code
import os
import boto3
import torch
import re
import pytorch_lightning
import torch.nn as nn
import torchvision.transforms as T
from lightning import Insta
from io import BytesIO

import base64
import PIL


S3 = boto3.client("s3")
s3 = boto3.resource("s3")


# def get_fake_event():
#     with open("2940801053738400977.jpg", "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     return {"body": encoded_string}


def inverse_transform(x):
    return 0.005 * (200 - 13 * x) / x / 6.8214285714 - 0.060775


def handler(event, context):
    base64_data = re.sub("^data:image/.+;base64,", "", event["body"])
    image_string = base64.b64decode(base64_data)
    filename = "/tmp/kuva"
    with open(filename, "wb+") as image_file:
        # image_to_s3("popin-data-bucket", filename, image_file)
        image_file.write(image_string)
    img = PIL.Image.open(filename)
    # img = image_from_s3("popin-data-bucket", filename)
    model = Predictor()
    result = model.forward(img)
    result = (float(inverse_transform(result[0][0])) / 1.5) * 10
    return {"statusCode": 200, "body": result}


def image_to_s3(bucket, key, data):
    object = s3.Object(bucket, key)
    object.put(Body=data)


def image_from_s3(bucket, key):
    bucket = s3.Bucket(bucket)
    image = bucket.Object(key)
    img_data = image.get().get("Body").read()
    return PIL.Image.open(BytesIO(img_data))


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


# event = get_fake_event()
# handler(event, None)
