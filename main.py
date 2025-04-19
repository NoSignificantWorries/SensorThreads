import time
import argparse
import logging
import logging.config
import threading
import queue
import os

import cv2


class General:
    stop_flag = threading.Event()
    data_buffers = {}
    data = {}


class Sensor:
    def get(self):
        raise NotImplementedError("Subprocess must implement method get()")


class SensorX(Sensor):
    next_id = 1

    def __init__(self, delay: float):
        self._delay = delay
        self._delta = 0
        self._id = SensorX.next_id
        SensorX.next_id += 1
        General.data_buffers[self._id] = queue.Queue(maxsize=2)
        General.data[self._id] = 0
    
    def get(self) -> int:
        time.sleep(self._delay)
        self._delta += 1
        return (self._id, self._delta)


class SensorCam(Sensor):
    def __init__(self, camera_descriptor: str, resolution: tuple[int, int]):
        self._resolution = resolution
        self.cap = cv2.VideoCapture(camera_descriptor)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        General.data_buffers[0] = queue.Queue(maxsize=2)
        General.data[0] = None

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video device: {camera_descriptor}")

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            return (0, None)
        resized_frame = cv2.resize(frame, self._resolution, cv2.INTER_CUBIC)
        return (0, resized_frame)
    

class WindowImage:
    def __init__(self, freqency: float, resolution: tuple[int, int]):
        self.freq = freqency
        self._resolution = resolution
        self.org = (int(self._resolution[0] * 0.85), int(self._resolution[1] * 0.92))
        self.msg_step = 30

    def __del__(self):
        cv2.destroyAllWindows()

    def show(self, sensors: dict):
        frame = sensors[0]
        if frame is None:
            raise RuntimeError("Unknown error occured. Can`t read new frame!")

        time.sleep(self.freq)
        org_y = self.org[1]
        for id in sensors.keys():
            if not id:
                continue
            cv2.putText(frame,
                        f"sensor_{id}: {sensors[id]}",
                        (self.org[0], org_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 0),
                        lineType=cv2.LINE_AA,
                        bottomLeftOrigin=False)
            org_y += self.msg_step

        cv2.imshow("Cam", frame)


def SensorThread(sensor: Sensor):
    global General
    while not General.stop_flag.is_set():
        id, val = sensor.get()
        General.data_buffers[id].put(val)


def main(device: str, resolution: str, freq: float) -> int:
    global General
    
    if not os.path.exists("log"):
        os.makedirs("log")
    
    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger("my_logger")
    
    if freq < 0:
        logger.critical(f"Current frequency ({freq}) less then zero, but it must be >= 0")
        raise ValueError("Wrong frequency value.")

    try:
        resolution = tuple(map(int, resolution.split("x")))
    except BaseException:
        logger.critical("Wrong resolution format, it should be <integer>x<integer>!")
        raise ValueError("Wrong resolution format.")
    
    if resolution[0] <= 0 or resolution[1] <= 0:
        logger.critical(f"Wrong dimension values width: {resolution[0]}, height: {resolution[1]}")
        raise ValueError("Wrong resolution format.")

    try:
        cam_sensor = SensorCam(device, resolution)
    except ValueError as err:
        logger.critical(err)
        raise ValueError("Errors with camera sensor.")
    
    sensor0 = SensorX(0.01)
    sensor1 = SensorX(0.1)
    sensor2 = SensorX(1)
    window = WindowImage(freq, resolution)

    cam_thread = threading.Thread(target=SensorThread, args=(cam_sensor,))
    sensor0_thread = threading.Thread(target=SensorThread, args=(sensor0,))
    sensor1_thread = threading.Thread(target=SensorThread, args=(sensor1,))
    sensor2_thread = threading.Thread(target=SensorThread, args=(sensor2,))
    
    cam_thread.daemon = True
    sensor0_thread.daemon = True
    sensor1_thread.daemon = True
    sensor2_thread.daemon = True
    
    cam_thread.start()
    sensor0_thread.start()
    sensor1_thread.start()
    sensor2_thread.start()

    try:
        time.sleep(1)
        while not General.stop_flag.is_set():
            for id in General.data_buffers.keys():
                if not General.data_buffers[id].empty():
                    General.data[id] = General.data_buffers[id].get()
            
            try:
                window.show(General.data)
            except RuntimeError as err:
                logger.critical(err)
                General.stop_flag.set()
                raise RuntimeError("Unknown errors with camera.")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                General.stop_flag.set()
                break

    except KeyboardInterrupt:
        General.stop_flag.set()
        print("Exit...")

    time.sleep(0.1)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--device", type=str, help="Name of your video device", default="/dev/video0")
    parser.add_argument("-r", "--resolution", type=str, help="Video resolution", default="1920x1440")
    parser.add_argument("-f", "--freq", type=float, help="Video flow frequency", default=0.033)

    args = parser.parse_args()
    
    try:
        main(args.device, args.resolution, args.freq)
    except BaseException as exp:
        print(f"Some errors occured: {exp}")
