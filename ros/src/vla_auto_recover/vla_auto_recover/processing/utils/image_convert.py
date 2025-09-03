import numpy as np
from sensor_msgs.msg import Image

_enc2dtype_channels = {
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
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

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header

DTYPE_CHANNEL2ENCODING = {
    ('uint8', 1): 'mono8',
    ('uint8', 3): 'bgr8', 
    ('uint8', 4): 'rgba8',
    ('uint16', 1): 'mono16',
}

def numpy_to_imgmsg(arr: np.ndarray,
                    encoding: str | None = None,
                    frame_id: str = '',
                    stamp = None
                    ) -> Image:
    """
    np.ndarray → sensor_msgs.msg.Image へ変換（行パディング無し想定）

    Parameters
    ----------
    arr : np.ndarray
        C-contiguous の画像配列 (H,W[,C])。
    encoding : str | None
        'bgr8' 等を指定。None なら自動判定（DTYPE_CHANNEL2ENCODING）。
    frame_id : str
        Header.frame_id に入れる座標系名。不要なら空文字で OK。
    stamp : builtin_interfaces.msg.Time | None
        Header.stamp を明示したい場合に渡す。None なら 0 初期化。

    Returns
    -------
    Image
        sensor_msgs 画像メッセージ。
    """
    if not arr.flags['C_CONTIGUOUS']:
        # view だけで渡すなら ascontiguousarray でコピーを作る
        arr = np.ascontiguousarray(arr)

    if arr.ndim == 2:           # (H,W) → グレースケール
        height, width = arr.shape
        channels = 1
    elif arr.ndim == 3:         # (H,W,C)
        height, width, channels = arr.shape
    else:
        raise ValueError('arr must have shape (H,W) or (H,W,C)')

    dtype_str = arr.dtype.name
    if encoding is None:
        try:
            encoding = DTYPE_CHANNEL2ENCODING[(dtype_str, channels)]
        except KeyError as exc:
            raise ValueError(
                f'encoding を自動判定できません: dtype={dtype_str}, '
                f'channels={channels}') from exc

    # メッセージ生成
    msg = Image()
    msg.height = height
    msg.width = width
    msg.encoding = encoding
    msg.is_bigendian = 0 if arr.dtype.byteorder in ('<', '=') else 1
    msg.step = width * channels * arr.dtype.itemsize
    # ROSメッセージのdataフィールドはbytes型を期待するため、tobytes()を使用
    msg.data = arr.tobytes()

    # Header（任意）
    hdr = Header()
    hdr.frame_id = frame_id
    if stamp is not None:
        hdr.stamp = stamp
    msg.header = hdr

    return msg
