from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenetv1 import MobileNetV1
import os


places_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../download_ckpts/places_teachers/"
)

places_model_dict = {
    # teachers
    "ResNet34": (
        resnet34,
        places_model_prefix + "resnet34_vanilla/ckpt_epoch_100.pth",
    ),
    "ResNet50": (
        resnet50,
        places_model_prefix + "resnet50_vanilla/ckpt_epoch_100.pth",
    ),



    
    # students
    "MobileNetV1": (MobileNetV1, None),
    "ResNet18": (resnet18, None),


}
