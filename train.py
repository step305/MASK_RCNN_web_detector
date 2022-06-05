import pixellib
from pixellib.custom_train import instance_custom_training


train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes=1, batch_size=1)
train_maskrcnn.load_pretrained_model("mask_rcnn_coco.h5")
train_maskrcnn.load_dataset("dataset")
train_maskrcnn.visualize_sample()
train_maskrcnn.train_model(num_epochs=300, augmentation=True,  path_trained_models="mask_rcnn_models")