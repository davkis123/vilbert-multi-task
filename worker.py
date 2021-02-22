import sys
import os
import torch
import yaml
import pickle

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
from vilbert.task_utils import LoadDatasetEval

import numpy as np
import matplotlib.pyplot as plt
import PIL

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from PIL import Image
import cv2
import argparse
import glob
from types import SimpleNamespace
import pdb


class FeatureExtractor:
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser()
        self.detection_model = self._build_detection_model()

    def get_parser(self):        
        parser = SimpleNamespace(model_file= 'data/detectron_model.pth',
                                config_file='data/detectron_config.yaml',
                                batch_size=1,
                                num_features=100,
                                feature_name="fc6",
                                confidence_threshold=0,
                                background=False,
                                partition=0)
        return parser
    
    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cpu")
        model.eval()
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)
        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)
            cls_prob = torch.max(scores[keep_boxes][start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    "cls_prob": scores[keep_boxes].cpu().numpy(),
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos = [], [], []

        for image_path in image_paths:
            im, im_scale, im_info = self._image_transform(image_path)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cpu")

        with torch.no_grad():
            output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = file_base_name + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), info)

    def extract_features(self, image_path):

        features, infos = self.get_detectron_features([image_path])

        return features, infos

def prediction(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, model):

    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, attn_data_list = model(
        question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, output_all_attention_masks=True
    )


    # vil_entaliment:  
    # label_map = {0:"contradiction", 1:"neutral", 2:"entailment"}
    # logtis_tri = torch.max(vil_tri_prediction, 1)[1].data
    # print("Entaliment: " + str(label_map[logtis_tri.item()]))

    label_map = {0:"contradiction (false)", 1:"neutral", 2:"entailment (true)"}
        
    # logtis_tri = torch.max(vil_tri_prediction, 1)[1].data
    prob_tri = torch.softmax(vil_tri_prediction.view(-1), dim=0)
    prob_val, prob_idx = torch.sort(prob_tri, 0, True)

    answer = [label_map[prob_idx[i].item()] for i in range(3)]
    confidence = [prob_val[i].item() for i in range(3)]
    output = {
        "top3_answer": answer,
        "top3_confidence": confidence
    }
    print(output)

def custom_prediction(query, features, infos, tokenizer, model, task = [13]):

    tokens = tokenizer.encode(query)

    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)

    max_length = 37
    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [0] * (max_length - len(tokens))
        tokens = tokens + padding
        input_mask += padding
        segment_ids += padding

    text = torch.from_numpy(np.array(tokens)).unsqueeze(0)
    input_mask = torch.from_numpy(np.array(input_mask)).unsqueeze(0)
    segment_ids = torch.from_numpy(np.array(segment_ids)).unsqueeze(0)
    task = torch.from_numpy(np.array(task)).unsqueeze(0)
    num_image = len(infos)

    feature_list = []
    image_location_list = []
    image_mask_list = []

    for i in range(num_image):
        image_w = infos[i]['image_width']
        image_h = infos[i]['image_height']
        feature = features[i]
        num_boxes = feature.shape[0]

        g_feat = torch.sum(feature, dim=0) / num_boxes
        num_boxes = num_boxes + 1
        feature = torch.cat([g_feat.view(1,-1), feature], dim=0)
        boxes = infos[i]['bbox']
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:,:4] = boxes
        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)
        g_location = np.array([0,0,1,1,1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        image_mask = [1] * (int(num_boxes))

        feature_list.append(feature)
        image_location_list.append(torch.tensor(image_location))
        image_mask_list.append(torch.tensor(image_mask))


    features = torch.stack(feature_list, dim=0).float()
    spatials = torch.stack(image_location_list, dim=0).float()
    image_mask = torch.stack(image_mask_list, dim=0).byte()
    co_attention_mask = torch.zeros((num_image, num_boxes, max_length))

    prediction(text, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task, model)

def load_vilbert_model():
    global feature_extractor
    global tokenizer
    global model
    # =============================
    # ViLBERT part
    # =============================


    feature_extractor = FeatureExtractor()

    args = SimpleNamespace(from_pretrained= "multi_task_model.bin",
                           bert_model="bert-base-uncased",
                           config_file="config/bert_base_6layer_6conect.json",
                           max_seq_length=101,
                           train_batch_size=1,
                           do_lower_case=True,
                           predict_feature=False,
                           seed=42,
                           num_workers=0,
                           baseline=False,
                           img_weight=1,
                           distributed=False,
                           objective=1,
                           visual_target=0,
                           dynamic_attention=False,
                           task_specific_tokens=True,
                           tasks='1',
                           save_name='',
                           in_memory=False,
                           batch_size=1,
                           local_rank=-1,
                           split='mteval',
                           clean_train_sets=True
                          )


    timeStamp = args.from_pretrained.split('/')[-1] + '-' + args.save_name
    config = BertConfig.from_json_file(args.config_file)
    default_gpu=True

    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.task_specific_tokens:
        config.task_specific_tokens = True    

    if args.dynamic_attention:
        config.dynamic_attention = True

    config.visualization = True
    num_labels = 3129

    if args.baseline:
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )
    else:
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )

    model.eval()
    cuda = torch.cuda.is_available()
    if cuda: model = model.cuda(0)
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    with open('model.pkl', 'wb') as handle:
      pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('tokenizer.pkl', 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
"""def callback(body):
    global tokenizer
    print("I'm callback")
    start = time.time()
    body = yaml.safe_load(body) # using yaml instead of json.loads since that unicodes the string in value
    print(" [x] Received %r" % body)
    try:
        image_path = body["image_path"]
        features, infos = feature_extractor.extract_features(image_path)
        query = body["question"]
        socket_id = body["socket_id"]
        
        answer = custom_prediction(query, features, infos, tokenizer)
        
        top3_answer = answer["top3_answer"]
        top3_confidence = answer["top3_confidence"]
        top3_list = []
        for i in range(3):
            temp = {}
            temp["answer"] = top3_answer[i]
            temp["confidence"] = round(top3_confidence[i]*100, 2)
            top3_list.append(temp)

        result = {
            "result": top3_list
        }
        print("The task result is", result)
        question_obj.answer_text = result
        question_obj.save()
    
    except Exception as e:
        print(traceback.print_exc())
        print(str(e))

    end = time.time()
    print("Time taken is", end - start)"""


def main():
    # Load correponding Vilbert model into global instance
    load_vilbert_model()

if __name__ == "__main__":
    main()