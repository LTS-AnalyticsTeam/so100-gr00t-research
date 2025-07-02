import numpy as np
from sensor_msgs.msg import Image

_enc2dtype_channels = {
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    # 必要に応じて追加
}


def imgmsg_to_ndarray(msg: Image) -> np.ndarray:
    try:
        dtype, ch = _enc2dtype_channels[msg.encoding]
    except KeyError as e:
        raise TypeError(f"Unsupported encoding: {msg.encoding}") from e

    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    buf = np.frombuffer(msg.data, dtype=dtype)

    if ch == 1:
        return buf.reshape(msg.height, msg.width)
    return buf.reshape(msg.height, msg.width, ch)
