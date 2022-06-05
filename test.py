import pixellib
from pixellib.custom_train import instance_custom_training
import os


if __name__ == '__main__':
    model_path = os.path.join(os.path.abspath(os.path.curdir), "mask_rcnn_models")
    train_maskrcnn = instance_custom_training()
    train_maskrcnn.modelConfig(network_backbone="resnet101", num_classes=1)
    train_maskrcnn.load_dataset("dataset")
    model_files = sorted([os.path.join(model_path, file_name) for file_name in os.listdir(model_path)])
    train_maskrcnn.evaluate_model(model_path)
