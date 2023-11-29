from pathlib import Path

import cv2
from numpy import ndarray, where

from util import prettify_timestamp

class Trackable:

    name: str

    def write_instances(self, target_dir: Path):
        for frame, frame_count in self.instances:
            print(f"Found {self.name} at {int(frame_count)}")
            # output_path = str((target_dir / f"{self.name}_{int(frame_count)}.png").absolute().as_posix())
            output_path = f"{self.name}_{int(frame_count)}.png"
            print(f"Writing to {output_path}")
            cv2.imwrite(output_path, frame)

    def __init__(self):
        pass

    @property
    def triggered(self):
        pass

    def process(self):
        pass

class FirstAppearance(Trackable):

    indicator: ndarray
    frame: None | ndarray
    frame_count: None | int
    threshold: float
    name: str

    def __init__(self, name, indicator, threshold=0.8):
        self.name = name
        self.indicator = indicator
        self.frame = None
        self.threshold = threshold

    def __str__(self):
        return f"{self.name}: appeared at #{self.frame_count}"

    @property
    def instances(self):
        if self.frame is None:
            yield (None, None)

        yield (self.frame, self.frame_count)

    @property
    def triggered(self):
        return self.frame is not None

    @property
    def triggered_at(self):
        return int(self.frame_count or 0)

    def get_timestamp(self, framerate, pretty=False):
        seconds = self.frame_count / framerate
        if pretty:
            return prettify_timestamp(seconds)

        return seconds

    def process(self, image, frame_count):
        if self.triggered:
            return

        target_array = cv2.matchTemplate(
            image,
            self.indicator,
            cv2.TM_CCOEFF_NORMED
        )

        if len(where( target_array >= self.threshold)[0]):
            self.frame = image
            self.frame_count = frame_count


class DependentAppearance(FirstAppearance):

    dependent_on: Trackable

    def __init__(self, name, indicator, dependent_on, threshold=0.8):
        super().__init__(name, indicator, threshold)
        self.dependent_on = dependent_on

    def process(self, image, frame):
        if not self.dependent_on.triggered:
            return

        super().process(image, frame)

class FirstDisappearance(Trackable):

    name: str
    indicator: ndarray

    appeared_frame: None | ndarray
    appeared_at: None | int

    disappeared_frame: None | ndarray
    disappeared_at: None | int
    threshold: float

    def __init__(self, name, indicator, threshold=0.8):
        self.name = name
        self.indicator = indicator

        self.appeared_frame = None
        self.appeared_at = None

        self.disappeared_frame = None
        self.disappeared_at = None

        self.threshold = threshold

    def __str__(self):
        return f"{self.name}: appeared at #{self.appeared_at}, disappeared at #{self.disappeared_at}"

    @property
    def instances(self):
        yield (self.appeared_frame, self.appeared_at)
        yield (self.disappeared_frame, self.disappeared_at)

    @property
    def triggered(self):
        return self.disappeared_at is not None

    @property
    def triggered_at(self):
        return int(self.disappeared_at or 0)

    def get_timestamp(self, framerate, pretty=False):
        seconds = self.disappeared_at / framerate
        if pretty:
            return prettify_timestamp(seconds)

        return seconds

    def process(self, image, frame):
        if self.triggered:
            return

        target_array = cv2.matchTemplate(
            image,
            self.indicator,
            cv2.TM_CCOEFF_NORMED
        )

        found = len(where( target_array >= self.threshold)[0])
        if self.appeared_at is None and found:
            self.appeared_frame = image
            self.appeared_at = frame
        elif self.appeared_at is not None and not found:
            self.disappeared_frame = image
            self.disappeared_at = frame


class LastDisappearance(Trackable):

    name: str
    indicator: ndarray

    appeared_frame: None | ndarray
    appeared_at: None | int
    on_screen: bool

    disappeared_frame: None | ndarray
    disappeared_at: None | int
    threshold: float

    def __init__(self, name, indicator, threshold=0.8):
        self.name = name
        self.indicator = indicator

        self.appeared_frame = None
        self.appeared_at = None
        self.on_screen = False

        self.disappeared_frame = None
        self.disappeared_at = None

        self.threshold = threshold

    def __str__(self):
        return f"{self.name}: first appeared at #{self.appeared_at}, last disappeared at #{self.disappeared_at}"

    @property
    def instances(self):
        yield (self.appeared_frame, self.appeared_at)
        yield (self.disappeared_frame, self.disappeared_at)

    @property
    def triggered(self):
        return self.disappeared_at is not None

    @property
    def triggered_at(self):
        return int(self.disappeared_at or 0)

    def get_timestamp(self, framerate, pretty=False):
        seconds = self.disappeared_at / framerate
        if pretty:
            return prettify_timestamp(seconds)

        return seconds

    def process(self, image, frame):
        target_array = cv2.matchTemplate(
            image,
            self.indicator,
            cv2.TM_CCOEFF_NORMED
        )

        found = len(where( target_array >= self.threshold)[0])
        if self.appeared_at is None and found:
            self.appeared_frame = image
            self.appeared_at = frame
            self.on_screen = True

        elif self.on_screen and not found:
            self.disappeared_frame = image
            self.disappeared_at = frame

        self.on_screen = found


class MultipleInstances(Trackable):

    indicator: ndarray
    appearances: list

    def __init__(self, indicator):
        self.indicator = indicator
        self.appearances = list()

    def process(self, image, frame):
        pass
