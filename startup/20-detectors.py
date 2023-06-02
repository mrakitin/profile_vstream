import datetime
import logging
import time as ttime
from collections import deque
from pathlib import Path

import cv2
import h5py
import numpy as np
from area_detector_handlers import HandlerBase
from event_model import compose_resource
from ophyd import Component as Cpt
from ophyd import Device, Signal
from ophyd.sim import NullStatus, new_uid

logger = logging.getLogger("vstream")


from nslsii.detectors.webcam import VideoStreamDet
from nsls2_detector_handlers.webcam import VideoStreamHDF5Handler

vstream = VideoStreamDet(
    video_stream_url="http://10.66.217.44/mjpg/video.mjpg", name="vstream",
    frame_shape=(1080, 1920),
)
vstream.exposure_time.put(0.5)

db.reg.register_handler("VIDEO_STREAM_HDF5", VideoStreamHDF5Handler, overwrite=True)

# Logger config:
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)
# handler.setLevel(logging.DEBUG)
