import pixellib
from pixellib.custom_train import instance_custom_training
import os
from pixellib.instance.utils import compute_ap
from pixellib.instance.mask_rcnn import mold_image
from pixellib.instance.mask_rcnn import MaskRCNN
from pixellib.instance.mask_rcnn import load_image_gt
import numpy as np


def calc_metrics(mask_rcnn_instance, model_file, iou_threshold=0.5):
    mask_rcnn_instance.model = MaskRCNN(mode="inference", model_dir=os.getcwd(), config=mask_rcnn_instance.config)

    if str(model_file).endswith(".h5"):
        mask_rcnn_instance.model.load_weights(model_file, by_name=True)
    APs = []
    precisions_all = []
    # outputs = list()
    for image_id in mask_rcnn_instance.dataset_test.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(mask_rcnn_instance.dataset_test,
                                                                         mask_rcnn_instance.config, image_id)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, mask_rcnn_instance.config)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = mask_rcnn_instance.model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"],
                                                       r["scores"], r['masks'],
                                                       iou_threshold=iou_threshold)
        # store
        APs.append(AP)
        precisions_all.append(precisions)
    # calculate the mean AP across all images
    mAP = np.mean(APs)
    return mAP, APs, precisions_all


if __name__ == '__main__':
    model_path = os.path.join(os.path.abspath(os.path.curdir), "mask_rcnn_models\\mask_rcnn_model.resnet101.h5")
    train_maskrcnn = instance_custom_training()
    train_maskrcnn.modelConfig(network_backbone="resnet101", num_classes=1)
    train_maskrcnn.load_dataset("dataset")
    # model_files = sorted([os.path.join(model_path, file_name) for file_name in os.listdir(model_path)])
    # train_maskrcnn.evaluate_model(model_path)
    result50 = calc_metrics(train_maskrcnn, model_file=model_path, iou_threshold=0.5)
    result75 = calc_metrics(train_maskrcnn, model_file=model_path, iou_threshold=0.75)
    print('mAP50 = ' + str(result50[0]))
    print('APs50 = ' + str(result50[1]))
    print('mAP75 = ' + str(result75[0]))
    print('APs75 = ' + str(result75[1]))
