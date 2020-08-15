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

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures import Boxes, Instances, BoxMode

# import some common libraries

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
parser.add_argument("--ignore_if_present", action="store_true",
                    help="If specified the script will ignore features files that are already present")
parser.add_argument("--is_gw", action="store_true", help="Defines whether we're processing GuessWhat?!")
args = parser.parse_args()


def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    class_distr_scores = scores.clone()
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
    num_objs = torch.arange(num_objs)
    if torch.cuda.is_available():
        num_objs = num_objs.cuda()
    idxs = num_objs * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]  # Select max boxes according to the max scores.

    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    class_distr_scores = class_distr_scores[keep]
    # we set the background probability to 0
    class_distr_scores[:, -1] = 0.0
    result.scores = class_distr_scores

    return result, keep


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
            box_ids = []

            for i, boxes_data in enumerate(given_boxes):
                boxes = []
                curr_box_ids = []

                for bid, bbox in boxes_data:
                    boxes.append(bbox)
                    curr_box_ids.append(bid)

                raw_boxes = Boxes(torch.tensor(boxes, device=images.tensor.device))

                raw_image = raw_images[i]
                # Remember that raw_image has shape [height, width, color_channel]
                raw_height, raw_width = raw_image.shape[:2]
                # Remember that images[i] has shape [color_channel, height, width]
                new_height, new_width = images[i].shape[1:]
                # Scale the box
                scale_x = 1. * new_width / raw_width
                scale_y = 1. * new_height / raw_height
                boxes = raw_boxes.clone()
                boxes.scale(scale_x=scale_x, scale_y=scale_y)
                proposal_boxes.append(boxes)
                original_boxes.append(raw_boxes)
                box_ids.append(curr_box_ids)

            features = [features[f] for f in detector.model.roi_heads.in_features]
            box_features = detector.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_proposal_deltas = detector.model.roi_heads.box_predictor(feature_pooled)
            pred_class_prob = torch.softmax(pred_class_logits, -1)
            # we reset the background class that we will ignore later on
            pred_class_prob[:, -1] = 0.0

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
                    scores=pred_class_prob[indexes],
                    features=roi_features[indexes],
                    box_ids=box_ids[batch_idx]
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
            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image_size,
                    nms_thresh=nms_thresh, topk_per_image=args.max_boxes
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
        img_paths, imgs, split_ids, img_ids, boxes = [], [], [], [], []

        for doc in pathXid_trunk:
            img_paths.append(doc["path"])
            img_ids.append(doc["image_id"])
            image = cv2.imread(doc["path"])
            if image is None:
                raise ValueError(f"Unable to read image {doc['path']}")
            imgs.append(image)
            split_ids.append(doc["split"])
            if "boxes" in doc:
                boxes.append(doc["boxes"])

        instances_list = extract_features(args, detector, imgs, boxes)

        for img, img_id, split, instances in zip(imgs, img_ids, split_ids, instances_list):
            instances = instances.to('cpu')
            features = instances.features

            num_objects = len(instances)

            item = {
                "img_id": img_id,
                "img_h": img.shape[0],
                "img_w": img.shape[1],
                "objects2id": instances.box_ids if hasattr(instances, "box_ids") else np.arange(num_objects),  # int64
                "objects_conf": instances.scores.numpy(),  # float32
                "num_boxes": num_objects,
                "boxes": instances.pred_boxes.tensor.numpy(),  # float32
                "features": features.numpy()  # float32
            }

            split_dir = os.path.join(args.output_dir, split)

            if not os.path.exists(split_dir):
                os.makedirs(split_dir)

            np.savez(
                os.path.join(split_dir, str(img_id)),
                **item
            )


def load_image_annotations(image_root, images_metadata, output_dir, use_gold_boxes=False, ignore_if_present=False,
                           is_gw=False):
    annotations = []

    for split, split_data in images_metadata.items():
        for image_data in split_data:
            if is_gw:
                image_path = os.path.join(image_root, image_data.get("file_name", f"{image_data['image_id']}.jpg"))
            else:
                image_path = os.path.join(image_root, split, f"{image_data['image_id']}.jpg")

            feature_path = os.path.join(output_dir, split, f"{image_data['image_id']}.npz")

            if ignore_if_present and os.path.exists(feature_path):
                continue

            ann = {
                "path": image_path,
                "image_id": image_data["image_id"],
                "split": split
            }

            if use_gold_boxes:
                ann["boxes"] = [(o["id"], BoxMode.convert(o["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)) for o in
                                image_data["objects"]]

            annotations.append(
                ann
            )

    return annotations


def build_model():
    cfg = get_cfg()  # Renew the cfg file
    cfg.merge_from_file("configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
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

    paths = load_image_annotations(args.image_root, images_metadata, args.output_dir, args.gold_boxes,
                                   args.ignore_if_present, args.is_gw)  # Get paths and ids
    print("-- Loading FastRCNN model...")
    detector = build_model()

    print(f"-- Extracting features for {len(paths)} images using batch size {args.batch_size}")
    extract_dataset_features(args, detector, paths)


if __name__ == "__main__":
    main(args)
