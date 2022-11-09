# vision-robotx-yolox

**TODO**
 - [x] Docker containers for local machines
 - [x] Detection model running
 - [x] Detection model server publishing to ZeroMQ
 - [x] Example client code for visualising boxes and images seperate from the server
 - [ ] Docker containers for Jetson Nano
 - [ ] Publish Jetson containers to a docker hub for pulling onto Jetson Nano
 - [ ] Have the Example server and client running on Jetson Nano
 - [ ] Have the Jetson nano automatically pull the newest container and start the server as a daemon. 


## Building the Docker containers

The docker containers are built using docker-compose, as follows:
```
docker-compose build yolox
```
This will build the container labelled as `'yolox'` in the `docker-compose.yml` file.

**NOTE**: Currently the container is buing built for an x86 machine with an Nvidia GPU.



## Running in Docker-Compose

You can create/connect to a bash terminal inside the built container using the following
```
docker-compose run --service-ports yolox bash
```


## Running Server

Once connected to the containers, you can run with the RobotX model as follows:
```
python3 inference_server.py  -f exps/default/robotx_nano.py --device gpu -c ./weights/robotx_weights-v1.pth --conf 0.25 --nms 0.45 --img_size 640
```

For testing, it is recommeneded to run with the default model. E.g People, Cars etc. 
```
python3 inference_server.py  -f exps/default/yolox_nano.py --device gpu -c ./weights/yolox_nano.pth --conf 0.25 --nms 0.45 --img_size 640 --classmap COCO
```

### Running Server with Image publishing

Adding the `--publish_images` flag will publish images to the topic set by `--img_topic`. This defaults to `'images'`

```
python3 inference_server.py  -f exps/default/yolox_nano.py --device gpu -c ./weights/yolox_nano.pth --conf 0.25 --nms 0.45 --img_size 640 --classmap COCO --publish_images
```

Images sent over zeromq are encoded using JPEG and the available code is located in `image_zmq.py`.


## Running the example client code

The example client code is located in inference_client.py.

```
python3 inference_client.py
```
This will automatically detect if there are images sent and will decode them.

The client can also be started using the docker-compose file as follows:
```
docker-compose run --service-ports yolox-client
```



## Testing with local interactive mode

Note: this is a bit older and will the client and server code in one process.
It is useful for testing.

```
python3 tools/interative_inference.py  -f exps/default/yolox_nano.py --device gpu -c ./weights/yolox_nano.pth --conf 0.25 --nms 0.45 --img_size 640 --classmap COCO --fp16
```


# JETSON NANO

## Running Server on Nano

To run the server on the nano, it requires you to either build the container (see below) or to pull the latest docker container image from the docker hub. It is recommended to pull the latest image as follows:
```
docker pull ...
```

Once you have the image, start the container and run as follows:

```
docker run --rm -it --gpus all --ipc=host -p 5001:5001 -v $(pwd):/workspace/ -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=$DISPLAY --device /dev/video0:/dev/video0  seal-nano bash
```

## Building docker on nano

NOTE: This is only fo Building the container
```
docker build -f ./Dockerfile.nano -t seal-nano:latest .
```

