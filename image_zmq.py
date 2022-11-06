import zmq
import numpy as np
import json
from json import JSONEncoder
import cv2
import base64

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def send_array(socket, A, arrayname="NoName", flags=0, copy=True, track=False):
    """
    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    Also sends array name for display with cv2.show(image)."""

    if not A.flags['C_CONTIGUOUS']:
        # Make it contiguous before sending
        A = np.ascontiguousarray(A)


    md = dict(
        arrayname=arrayname,
        dtype=str(A.dtype),
        shape=A.shape,
        data = A
    )
    encoded_msg_data = json.dumps(md, cls=NumpyArrayEncoder)
    # print(type(encoded_msg_data), encoded_msg_data)
    return socket.send_string(encoded_msg_data, flags)
    
    # socket.send_json(md, flags | zmq.SNDMORE)
    # return socket.send(A, flags, copy=copy, track=track)


def recv_array(socket, flags=0, copy=True, track=False):
    """
    recv_array receives dict(arrayname,dtype,shape) and an array
    and reconstructs the array with the correct shape and array name.
    """
    md = socket.recv_string(flags=flags)
    decoded_msg = json.loads(md)
    print(decoded_msg.keys())
    # msg = socket.recv(flags=flags)
    # msg = socket.recv()
    # print(md)
    # buf = memoryview(msg)
    array = np.asarray(decoded_msg["data"])
    print(array.shape)
    return decoded_msg
    #msg = socket.recv(flags=flags, copy=copy, track=track)
    #A = np.frombuffer(msg, dtype=md['dtype'])
    # return (md['arrayname'], A.reshape(md['shape']))


def send_image(socket, img, arrayname="NoName", flags=0, copy=True, track=False):
    """
    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    Also sends array name for display with cv2.show(image)."""

    if not img.flags['C_CONTIGUOUS']:
        # Make it contiguous before sending
        A = np.ascontiguousarray(A)

    assert len(img.shape) == 3
    assert img.shape[-1] == 3

    encoded, buffer = cv2.imencode('.jpg', img)
    return socket.send(base64.b64encode(buffer), flags, copy=copy)   


def recv_image(socket):
    """
    recv_image receives a jpeg encoded image
    and reconstructs the array with the correct shape.
    """

    frame = socket.recv()
    img = base64.b64decode(frame)
    npimg = np.fromstring(img, dtype=np.uint8)
    image = cv2.imdecode(npimg, 1)
    return image