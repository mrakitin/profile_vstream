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


class ExternalFileReference(Signal):
    """
    A pure software Signal that describe()s an image in an external file.
    """

    def describe(self):
        resource_document_data = super().describe()
        resource_document_data[self.name].update(
            dict(
                external="FILESTORE:",
                dtype="array",
            )
        )
        return resource_document_data


class VideoStreamDet(Device):
    image = Cpt(ExternalFileReference, kind="normal")
    exposure_period = Cpt(Signal, value=1.0, kind="config")

    def __init__(
        self,
        *args,
        root_dir="/tmp/video-stream-data",
        assets_dir=None,
        video_stream_url=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._root_dir = root_dir
        self._assets_dir = assets_dir
        self._video_stream_url = video_stream_url

        self._asset_docs_cache = deque()
        self._resource_document = None
        self._datum_factory = None

    def trigger(self, *args, **kwargs):
        super().trigger(*args, **kwargs)

        date = datetime.datetime.now()
        self._assets_dir = date.strftime("%Y/%m/%d")
        data_file = f"{new_uid()}.h5"

        self._resource_document, self._datum_factory, _ = compose_resource(
            start={"uid": "needed for compose_resource() but will be discarded"},
            spec="VIDEO_STREAM_HDF5",
            root=self._root_dir,
            resource_path=str(Path(self._assets_dir) / Path(data_file)),
            resource_kwargs={},
        )
        # now discard the start uid, a real one will be added later
        self._resource_document.pop("run_start")
        self._asset_docs_cache.append(("resource", self._resource_document))

        data_file = str(
            Path(self._resource_document["root"])
            / Path(self._resource_document["resource_path"])
        )

        frames = []
        times = []

        start = ttime.monotonic()
        i = 0
        cap = cv2.VideoCapture(self._video_stream_url)
        while True:
            logger.debug(f"Iteration: {i}")
            i += 1
            ret, frame = cap.read()
            frames.append(frame)
            times.append(ttime.time())

            # cv2.imshow('Video', frame)
            logger.debug(f"shape: {frame.shape}")

            if ttime.monotonic() - start >= self.exposure_period.get():
                break

            if cv2.waitKey(1) == 27:
                exit(0)

        frames = np.array(frames)
        logger.debug(f"original shape: {frames.shape}")
        # Averaging over all frames and summing 3 RGB channels
        averaged = frames.mean(axis=0).sum(axis=-1)

        with h5py.File(data_file, "w") as f:
            group = f.create_group("/entry")
            group.create_dataset("frames", data=frames, compression="lzf")
            group.create_dataset("averaged", data=averaged, compression="lzf")

        datum_document = self._datum_factory(datum_kwargs={})
        self._asset_docs_cache.append(("datum", datum_document))

        self.image.put(datum_document["datum_id"])

        self._resource_document = None
        self._datum_factory = None

        return NullStatus()

    def describe(self):
        res = super().describe()
        res[self.image.name].update(dict(shape=[480, 704]))
        return res

    def unstage(self):
        super().unstage()
        self._resource_document = None

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        for item in items:
            yield item


vstream = VideoStreamDet(
    video_stream_url="http://10.68.57.34/mjpg/video.mjpg", name="vstream"
)
vstream.exposure_period.put(0.5)


class VideoStreamHDF5Handler(HandlerBase):
    specs = {"VIDEO_STREAM_HDF5"}

    def __init__(self, filename):
        self._name = filename

    def __call__(self):
        with h5py.File(self._name, "r") as f:
            entry = f["/entry/averaged"]
            return entry[:]


db.reg.register_handler("VIDEO_STREAM_HDF5", VideoStreamHDF5Handler, overwrite=True)

# Logger config:
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)
# handler.setLevel(logging.DEBUG)
