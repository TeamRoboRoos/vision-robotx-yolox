# vision-robotx-yolox


## Running Server


Run with the RobotX model.
```
python3 inference_server.py  -f exps/default/robotx_nano.py --device gpu -c ./weights/robotx_weights-v1.pth --conf 0.25 --nms 0.45 --img_size 640
```


Run with the default model. E.g People, Cars etc. 
This model is good for testing.
```
python3 inference_server.py  -f exps/default/yolox_nano.py --device gpu -c ./weights/yolox_nano.pth --conf 0.25 --nms 0.45 --img_size 640 --classmap COCO
```

