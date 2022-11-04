import time
import os
import torch
import cv2
from argparse import ArgumentParser
import zmq

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from loguru import logger


def build_arg_parser():
    parser = ArgumentParser("RobotX Vision System Inference")

    # Netowkring arguments
    parser.add_argument(
        "--host", default="*", help="The host IP. Defaults to *"
    )

    parser.add_argument(
        "--publish_port", default="5001", help="The Port for messages to be published on."
    )
    
    parser.add_argument(
        "--bbox_topic", default="bboxes", help="The topic to publish bbox messages on."
    )

    parser.add_argument(
        "--img_topic", default="images", help="The topic to publish image messages on."
    )

    parser.add_argument(
        "--save_result",
        action="store_true",
        help="Saves the inference result of the video/stream if set.",
    )
    return parser


def visualise(cls_names, output, img_info, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return img
    output = output.cpu()

    bboxes = output[:, 0:4]

    # preprocessing: resize
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis(img, bboxes, scores, cls, cls_conf, cls_names)
    return vis_res




def main(args):
    
    host = "127.0.0.1"
    port = "5001"

    # Creates a socket instance
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    # Connects to a bound socket
    socket.connect("tcp://{}:{}".format(host, port))

    # Subscribes to all topics
    socket.subscribe("")

    # Receives a string format message
    while True:
        # First recieve the Topic
        topic = socket.recv_string()
        
        # Then recieve the json blob
        messagedata = socket.recv_json()
        print (topic, messagedata['num_bboxes'])


        # TODO: Have the server send imgs and client to visualise
        # TODO: THis means looking at multi topic recv
        
        # cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
        # cv2.imshow("yolox", result_frame)
        # ch = cv2.waitKey(1)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)