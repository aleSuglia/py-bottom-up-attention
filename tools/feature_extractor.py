# coding=utf-8
# Copyleft 2019 Project LXRT

import argparse
import json
import os

import cv2
import numpy as np
import torch
import tqdm
from torchvision.ops import nms

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures import Boxes, Instances

# import some common libraries

D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__))  # Root of detectron2

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str,
                    help="Directory containing the dataset images")
parser.add_argument("--images_metadata", type=str,
                    help="JSON file containing the mapping between dataset split and images")
parser.add_argument("--output_dir", type=str, help="Output directory where the features will be stored in")
parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
parser.add_argument("--min_boxes", type=int, help="Minimum number of detected boxes", default=36)
parser.add_argument("--max_boxes", type=int, help="Maximum number of detected boxes", default=36)
parser.add_argument("--score_threshold", type=float, help="Score threshold for the bounding box detection", default=0.2)
parser.add_argument("--gold_boxes", action="store_true",
                    help="Specify if you want to use gold bounding boxes instead of region proposals")
args = parser.parse_args()


def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Select max scores
    max_scores, max_classes = scores.max(1)  # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs) * num_bbox_reg_classes + max_classes
    if torch.cuda.is_available():
        idxs = idxs.cuda()
    max_boxes = boxes[idxs]  # Select max boxes according to the max scores.

    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = max_classes[keep]

    return result, keep


def extract_features_given_boxes(predictor, raw_image, raw_boxes):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])

        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        # print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)

        # ----
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        print(pred_class_logits.shape)
        pred_class_prob = torch.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)

        # Detectron2 Formatting (for visualization only)
        roi_features = feature_pooled
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes
        )

        return instances, roi_features


def extract_features(args, detector, raw_images, given_boxes=None):
    with torch.no_grad():
        inputs = []

        for raw_image in raw_images:
            image = detector.transform_gen.get_transform(raw_image).apply_image(raw_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": raw_image.shape[0], "width": raw_image.shape[1]})
        images = detector.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = detector.model.backbone(images.tensor)

        # Feature extraction given the bounding boxes
        if given_boxes:
            # Process Boxes in batch mode
            proposal_boxes = []
            original_boxes = []

            for i, boxes in enumerate(given_boxes):
                raw_boxes = Boxes(torch.tensor(boxes))
                raw_image = raw_images[i]
                raw_height, raw_width = raw_image.shape[:2]
                # Scale the box
                new_height, new_width = image.shape[:2]
                scale_x = 1. * new_width / raw_width
                scale_y = 1. * new_height / raw_height
                boxes = raw_boxes.clone()
                boxes.scale(scale_x=scale_x, scale_y=scale_y)
                proposal_boxes.append(boxes)
                original_boxes.append(raw_boxes)

            features = [features[f] for f in detector.model.roi_heads.in_features]
            box_features = detector.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_proposal_deltas = detector.model.roi_heads.box_predictor(feature_pooled)
            pred_class_prob = torch.softmax(pred_class_logits, -1)
            pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)

            # Detectron2 Formatting (for visualization only)
            roi_features = feature_pooled

            outputs = []
            total_boxes = 0

            # roi_features.shape = (num_total_boxes, 2048)
            # we need to group the boxes by image id
            for batch_idx, raw_image in enumerate(raw_images):
                indexes = slice(total_boxes, total_boxes + len(given_boxes[batch_idx]))
                instances = Instances(
                    image_size=raw_image.shape[:2],
                    pred_boxes=original_boxes[batch_idx],
                    scores=pred_scores[indexes],
                    pred_classes=pred_classes[indexes],
                    features=roi_features[indexes]
                )

                outputs.append(instances)
                total_boxes += len(given_boxes[batch_idx])

            return outputs

        # Feature extraction without bounding boxes
        # Generate proposals with RPN
        proposals, _ = detector.model.proposal_generator(images, features, None)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in detector.model.roi_heads.in_features]
        box_features = detector.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # (sum_proposals, 2048), pooled to 1x1

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = detector.model.roi_heads.box_predictor(feature_pooled)
        rcnn_outputs = FastRCNNOutputs(
            detector.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            detector.model.roi_heads.smooth_l1_beta,
        )

        # Fixed-number NMS
        instances_list, ids_list = [], []
        probs_list = rcnn_outputs.predict_probs()
        boxes_list = rcnn_outputs.predict_boxes()
        for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
            for nms_thresh in np.arange(0.3, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image_size,
                    score_thresh=args.score_threshold, nms_thresh=nms_thresh, topk_per_image=args.max_boxes
                )
                if len(ids) >= args.min_boxes:
                    break
            instances_list.append(instances)
            ids_list.append(ids)

        # Post processing for features
        features_list = feature_pooled.split(
            rcnn_outputs.num_preds_per_image)  # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        roi_features_list = []
        for ids, features in zip(ids_list, features_list):
            roi_features_list.append(features[ids].detach())

        # Post processing for bounding boxes (rescale to raw_image)
        raw_instances_list = []
        for batch_idx, (instances, input_per_image, image_size) in enumerate(zip(
                instances_list, inputs, images.image_sizes
        )):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            raw_instances = detector_postprocess(instances, height, width)
            raw_instances.features = roi_features_list[batch_idx]
            raw_instances_list.append(raw_instances)

        return raw_instances_list


def extract_dataset_features(args, detector, paths):
    for start in tqdm.tqdm(range(0, len(paths), args.batch_size)):
        pathXid_trunk = paths[start: start + args.batch_size]
        img_paths, imgs, img_ids, boxes = [], [], [], []

        for doc in pathXid_trunk:
            img_paths.append(doc["path"])
            img_ids.append(doc["id"])
            imgs.append(cv2.imread(doc["path"]))
            if "boxes" in doc:
                boxes.append(doc["boxes"])

        instances_list = extract_features(args, detector, imgs, boxes)

        for img, img_id, instances in zip(imgs, img_ids, instances_list):
            instances = instances.to('cpu')
            features = instances.features

            num_objects = len(instances)

            item = {
                "img_id": img_id,
                "img_h": img.shape[0],
                "img_w": img.shape[1],
                "objects_id": instances.pred_classes.numpy(),  # int64
                "objects_conf": instances.scores.numpy(),  # float32
                "num_boxes": num_objects,
                "boxes": instances.pred_boxes.tensor.numpy(),  # float32
                "features": features.numpy()  # float32
            }

            np.savez(
                os.path.join(args.output_dir, str(img_id)),
                **item
            )


def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))


def load_image_annotations(image_root, images_metadata, use_gold_boxes):
    annotations = []

    for split, split_data in images_metadata.items():
        for image_data in split_data:
            ann = {
                "path": os.path.join(image_root, image_data.get("file_name", f"{image_data['image_id']}.jpg")),
                "id": image_data["image_id"]
            }

            if use_gold_boxes:
                ann["boxes"] = [bbox_xywh_to_xyxy(o["bbox"]) for o in image_data["objects"]]

            annotations.append(
                ann
            )

    return annotations


def build_model():
    cfg = get_cfg()  # Renew the cfg file
    cfg.merge_from_file(os.path.join(
        D2_ROOT, "configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml"))
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    detector = DefaultPredictor(cfg)
    return detector


def main(args):
    with open(args.images_metadata) as in_file:
        images_metadata = json.load(in_file)

    if not os.path.exists(args.output_dir):
        print(f"Directory {args.output_dir} doesn't exist. Creating it...")
        os.makedirs(args.output_dir)

    paths = load_image_annotations(args.image_root, images_metadata, args.gold_boxes)  # Get paths and ids
    print("-- Loading FastRCNN model...")
    detector = build_model()

    print(f"-- Extracting features for {len(paths)} images using batch size {args.batch_size}")
    extract_dataset_features(args, detector, paths)


if __name__ == "__main__":
    main(args)
