import time
import os
import torch
import cv2
import zmq

from image_zmq import send_array, send_image

from argparse import ArgumentParser
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from loguru import logger

ROBOTX_CLASSES = ("Boat", "Buoy - Other", "Circle Buoy - Black",
                  "Field Boundary Buoy - Orange", "Floating Dock",
                  "Gate Buoy - Green", "Gate Buoy - Red", "Gate Buoy - White",
                  "Kayak", "Light Tower", "Obstacle Buoy - Black", "Person")


def build_arg_parser():
    parser = ArgumentParser("RobotX Vision System Inference")

    parser.add_argument("--mode",
                        default="camera",
                        help="Model type, eg. video or camera")
    parser.add_argument(
        "--classmap",
        default="ROBOTX",
        choices=["COCO", "ROBOTX"],
        type=str,
        help="Sets the model categories to one of the provided choices")

    parser.add_argument("-e", "--experiment_name", type=str, default=None)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default=None,
                        help="model name")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help=
        "YOLOX Experiment description file. This is usually a python file.",
    )

    parser.add_argument(
        "--path",
        default=None,
        help="Path to the input video. Required if --mode is set to video")

    parser.add_argument("--camid",
                        type=int,
                        default=0,
                        help="Camera device id")

    parser.add_argument("-c",
                        "--ckpt",
                        default=None,
                        type=str,
                        help="Checkpoint for evaluation")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--conf",
        default=0.3,
        type=float,
        help=
        "Confidence threshold. Boxes with probabilities lower than this value will be filtered out."
    )
    parser.add_argument(
        "--nms",
        default=0.3,
        type=float,
        help=
        "Non-maximal supression threshold. Boxes with overlap more than the threshold are filtered."
    )
    parser.add_argument(
        "--img_size",
        default=None,
        type=int,
        help="The resolution of the image to run through the nextwork.")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Run in Float16 for improved inference latency",
    )

    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse layers (conv, batch norm). Should improve performance.",
    )

    parser.add_argument(
        "--save_result",
        action="store_true",
        help="Saves the inference result of the video/stream if set.",
    )

    # Netowkring arguments
    parser.add_argument("--host",
                        default="*",
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
        "--publish_images",
        default=False,
        action="store_true",
        help="Whether to publish images to the image channel.",
    )

    return parser


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=ROBOTX_CLASSES,
        device="cpu",
        fp16=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=False)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0],
                    self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(outputs,
                                  self.num_classes,
                                  self.confthre,
                                  self.nmsthre,
                                  class_agnostic=True)
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visualise(self, output, img_info, cls_conf=0.35):
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

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def build_bbox_msg(outputs, img_info, cls_names):
    def format_bbox(bbox):
        return {
            'x1': bbox[0].item(),
            'y1': bbox[1].item(),
            'x2': bbox[2].item(),
            'y2': bbox[3].item(),
            'prob':
            (bbox[4] *
             bbox[5]).item(),  # I think this is object prob * class prob
            'object_class_id': int(bbox[6].item()),
            'object_class_name': cls_names[int(bbox[6].item())],
            'img_width': img_info['width'],
            'img_height': img_info['height']
        }

    bboxes = []
    if outputs[0] is not None:
        for res in outputs[0]:
            # outputs[0] as there is only 1 image being proccessed at a time
            bbox = format_bbox(res.cpu())
            bboxes.append(bbox)

    return {
        "bboxes": bboxes,
        'num_bboxes': len(bboxes),
        'class_names': cls_names
    }


def run_inference(predictor, vis_folder, current_time, args):

    logger.info(f"Setting up Publisher")
    # Setup Networking
    context = zmq.Context()
    socket = context.socket(zmq.PUB)

    address = "tcp://{}:{}".format(args.host, args.publish_port)
    logger.info(f"Binding Socket to: {address}")

    # Binds the socket to a predefined port on localhost
    socket.bind(address)

    cap = cv2.VideoCapture(args.path if args.mode == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        os.makedirs(save_folder, exist_ok=True)
        if args.mode == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(save_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (int(width), int(height)))

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            json_msg = build_bbox_msg(outputs,
                                      img_info,
                                      cls_names=predictor.cls_names)
            #logger.info(f"Sending: {json_msg}")

            # First send the topic string with the flag to do send multpart.
            socket.send_string(args.bbox_topic, flags=zmq.SNDMORE)

            # Then send the data. As it doesnt have zmq.SNDMORE, this will terminate the recv.
            socket.send_json(json_msg)

            if args.publish_images:
                logger.info(f"Sending Image: {frame.shape}")
                socket.send_string(args.img_topic, flags=zmq.SNDMORE)
                send_image(socket, img=frame, copy=False)

        else:
            break


def main(args):

    if args.mode == "video" and args.path is None:
        logger.error("Missing argument 'path' when using --mode == video")

    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    CLASSES = ROBOTX_CLASSES if args.classmap == "ROBOTX" else COCO_CLASSES
    if not exp.num_classes == len(CLASSES):
        logger.error(
            f"Experiment/Model has the wrong number of classes for classmap: {args.classmap}"
        )
        raise Exception("Model class size mismatch")

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "visual_results")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.img_size is not None:
        exp.test_size = (args.img_size, args.img_size)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(
        model, exp.test_size)))

    # Setup the runtime device
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16

    # Set to eval to make the model deterministic
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    logger.info(f"Loading Checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("Checkpoint Loaded")

    if args.fuse:
        logger.info("\tFusing model")
        model = fuse_model(model)
        logger.info("\tFusing complete")

    predictor = Predictor(model, exp, CLASSES, args.device, args.fp16)

    current_time = time.localtime()

    run_inference(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)