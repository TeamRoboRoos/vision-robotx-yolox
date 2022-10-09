#!/bin/bash
docker run --rm -it --network host --privileged -v $(pwd):/workspace --device /dev/video0:/dev/video0 seal-ros2 /bin/bash
