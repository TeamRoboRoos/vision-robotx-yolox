# C ZeroMQ Example CLient for the RoboSeal Vision System

## Requirements

Install the requirements on an Ubuntu/Deb based system as follows:
```
RUN apt-get install -y libzmq3-dev libjansson-dev gcc g++ make
```

## Build
```
make all
```

## Run

For the client to work, the server needs to be running. 
See the top level README on how to start that server properly.

Run the server from a docker container or virtual environment with the following command:
NOTE: This model is the standard model which is trained on every day objects.
```
python3 inference_server.py  -f exps/default/yolox_nano.py --device gpu -c ./weights/yolox_nano.pth --conf 0.25 --nms 0.45 --img_size 640 --classmap COCO
```

Run the example_client with one of the following commands:
```
make run
```
or

```
./example_client
```
## Cleanup
To Clean the files
```
make clean
```