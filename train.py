import argparse
from ultralytics import YOLO

def main(opt):
    model = YOLO(opt.cfg)
    model.train(data=opt.data, epochs=opt.epochs, batch=opt.batch_size, imgsz=opt.img_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--data', type=str, default='dataset/data.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default='yolov8s.yaml', help='model config path')
    opt = parser.parse_args()
    main(opt)
