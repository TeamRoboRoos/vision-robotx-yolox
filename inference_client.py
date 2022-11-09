import time
import os
import torch
import cv2
from argparse import ArgumentParser
import zmq
from image_zmq import recv_array, recv_image

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from loguru import logger


def build_arg_parser():
    parser = ArgumentParser("RobotX Vision System Inference")

    # Netowkring arguments
    parser.add_argument("--host",
                        default="127.0.0.1",
                        help="The host IP. Defaults to *")

    parser.add_argument("--publish_port",
                        default="5001",
                        help="The Port for messages to be published on.")

    parser.add_argument("--bbox_topic",
                        default="bboxes",
                        help="The topic to publish bbox messages on.")

    parser.add_argument("--img_topic",
                        default="images",
                        help="The topic to publish image messages on.")

    parser.add_argument(
        "--save_result",
        action="store_true",
        help="Saves the inference result of the video/stream if set.",
    )
    return parser

   

def visualise(cls_names, json_bboxes, img, cls_conf=0.35, ratio=1):
    
    bboxes = []
    cls = []
    scores = [] 
    
    if json_bboxes is None:
        return img
    for bb in json_bboxes:
        bboxes.append(
            [bb['x1'], bb['y1'], bb['x2'], bb['y2']]
        )
        cls.append(
            [bb['object_class_id']]
        )
        scores.append([bb['prob']])

    bboxes = torch.FloatTensor(bboxes)
    cls = torch.FloatTensor(cls).squeeze(1)
    scores = torch.FloatTensor(scores).squeeze(1)
   
    # preprocessing: resize
    # bboxes /= ratio
    
    vis_res = vis(img, bboxes, scores, cls, cls_conf, cls_names)
    return vis_res
    # return img

def main(args):

    host = args.host
    port = "5001"

    # Creates a socket instance
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    address = "tcp://{}:{}".format(host, port)
    # Connects to a bound socket
    logger.info(f"Connecting to {address}")
    socket.connect(address)
    logger.info(f"Connected to {address}")

    # Subscribes to all topics
    socket.subscribe("")
    logger.info(f"Subscribed to {address}")

    image, bboxes, classes = None, None, None
    
    while True:
        try:
    
            # First recieve the Topic
            topic = socket.recv_string()

            if topic == "bboxes":
                # Then recieve the json blob
                msg = socket.recv_json()
                logger.info(f"BBoxes Msg: {topic}, Num Boxes - {msg['num_bboxes']}")

                bboxes = msg['bboxes']
                classes = msg['class_names']
            if topic == "images":
                image = recv_image(socket)
                logger.info(f"Image Msg: {topic}, {image.shape}")
                
                
            if image is not None and bboxes is not None:    
                res_image = visualise(classes, bboxes, image)
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", res_image)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break
       
        

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)