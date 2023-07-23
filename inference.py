import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )

    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )

    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


def main(args):
    # load model
    model = FastSAM(args.model_path)
    input = Image.open(args.img_path)
    input = input.convert("RGB")
    everything_results = model(
        input,
        device="cpu",
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou    
        )

    print(everything_results[0].masks.data.size())



if __name__ == "__main__":
    args = parse_args()
    main(args)
