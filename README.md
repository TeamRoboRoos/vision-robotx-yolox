# vision-robotx-yolox

**TODO**
 - [x] Docker containers for local machines
 - [x] Detection model running
 - [x] Detection model server publishing to ZeroMQ
 - [x] Example client code for visualising boxes and images seperate from the server
 - [x] Docker containers for Jetson Nano
 - [x] Publish Jetson containers to a docker hub for pulling onto Jetson Nano
 - [x] Have the Example server and client running on Jetson Nano
 - [x] Have the Jetson nano automatically pull the newest container and start the server as a daemon. 


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
python3 inference_server.py  -f exps/default/robotx_nano.py --device gpu --fp16 -c ./weights/robotx_weights-v1.pth --conf 0.25 --nms 0.45 --img_size 640
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
docker pull adrianjohnstonaus/roboseals-nano-server:latest
```

Once you have the image, start the container using the helper script and run as follows:

```
./start-nano-server.sh -v $(pwd):/workspace/ --run python3 inference_server.py  -f exps/default/robotx_nano.py --device gpu --fp16 -c ./weights/robotx_weights-v1.pth --conf 0.25 --nms 0.45 --img_size 640
```
This will ask for the sudo password as it requires accessing / passing through some devices.
NOTE: `-v $(pwd):/workspace/` this flag will overwrite the local version of the code and expects you to be in the top level directory of this repository. If you remove this flag, it will default to using the embedded version of the code which was stored in the docker image/container.


If you have permissions issues, make sure the script has the execution permmissions as follows:
```
chmod +x ./start-nano-server.sh
```

### Running the test/default model
This command runs the server for the standard model e.g. Trained on everyday objects. It is useful for testing. 
```
./start-nano-server.sh -v $(pwd):/workspace/ --run python3 inference_server.py  -f exps/default/yolox_nano.py --device gpu --fp16 -c ./weights/yolox_nano.pth --conf 0.25 --nms 0.45 --img_size 640 --classmap COCO
```

## Building docker on nano

NOTE: This is only fo Building the container
```
docker build -f ./Dockerfile.nano -t adrianjohnstonaus/roboseals-nano-server:latest .
```

Push the newly build container to the docker hub registry:
```
docker push adrianjohnstonaus/roboseals-nano-server:latest
```



## Notes on the start-nano-server.sh

This script is used to make executing the docker command easy. 
It handles passing X11 and required environment variables needed to visualise the outputs on the device (useful for testing). 

It first forwards the xhost:
```
sudo xhost +si:localuser:root
```
Then enables SSH X11 fowarding:
```
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod 777 $XAUTH
```

Finally it then runs the command:
```
sudo docker run --runtime nvidia -it --rm --network host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH \
    --privileged \
    --device /dev/video0:/dev/video0 \
    -p 5001:5001 \
    $USER_VOLUME $CONTAINER_IMAGE $USER_COMMAND
```

Where `$USER_VOLUME`, `$CONTAINER_IMAGE`, `$USER_COMMAND` can be passed in to overwrite defaults.

If `$USER_COMMAND` is empty, it will default to the container default. 

### Updating the mounted camera device

In the above command, it is possible to change the device mounting by changing:
```
--device /dev/video0:/dev/video0
```
to something like:
```
--device /dev/video1:/dev/video0
```

This will instead mount `/dev/video1` to the container devices of `/dev/video0`

