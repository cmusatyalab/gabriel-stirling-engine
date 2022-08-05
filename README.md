# Gabriel Stirling Engine

A wearable cognitive assistance application for a Stirling engine. You can buy
the kit to use this application
[here](https://www.amazon.com/DjuiinoStar-Hot-Stirling-Engine-Assembly/dp/B07PMBPZFV).

[Demo Video](https://youtu.be/tU8jyDh_DGs)

Images are processed using the TensorFlow
[implementation](https://github.com/tensorflow/models/blob/aa3e639f80c2967504310b0f578f0f00063a8aff/research/object_detection/meta_architectures/faster_rcnn_meta_arch.py)
of Faster R-CNN. Cropped bouding boxes are classified using
[fast MPN-COV](https://github.com/jiangtaoxie/fast-MPN-COV).

## Server Installation

This code requires an Nvidia GPU. You must also have the
[Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
installed.

1. Download the
   [classifiers](https://owncloud.cmusatyalab.org/owncloud/index.php/s/n7g0YpP3csnUfV7)
   to a directory on the machine you plan to run this code from.
   Download the
   [object detectors](https://owncloud.cmusatyalab.org/owncloud/index.php/s/NyxPMaJvlY0AQRX)
   to a different directory on the same machine.
2. Follow the
   [instructions](https://github.com/tensorflow/models/blob/aa3e639f80c2967504310b0f578f0f00063a8aff/research/object_detection/g3doc/tf2.md#installation)
   for Docker Installation of the TensorFlow Object Detection API.
3. Start the container with the command
   `docker run --rm -it -p 9099:9099 -v /path/to/classifiers:/classifiers -v /path/to/object_detectors:/object_detectors -v /path/to/this/repo:/stirling --gpus all od`,
   where `/path/to/classifiers` is the directory that you downloaded the
   classifiers to and `/path/to/this/repo` is the location of this repository on
   your local filesystem.
4. Install dependencies by running
   `python3 -m pip install -r /stirling/server/requirements.txt`
5. Commit this repository using `docker commit`.

## Usage

Within the container, run `python3 /stirling/server/phone.py` to run the server
for the Android phone client, or run `python3 /stirling/server/glass.py` to run
the server for the Google glass client.

Both clients can be opened as a project using Android Studio. The client for an
Android phone is in the `phone-app` directory, and the client for a Google Glass
is in the `glass-app` directory. Add the line `gabrielHost="THE_SERVER_HOST"` to
`local.properties` for the respective client. The Google Glass client has been
tested on a
Google Glass Enterprise Edition 2. It will not work on the earlier Explorer
Edition.

The Android phone client allows you to zoom the camera image py performing a
pinch gesture on the main viewfinder. The Glass client does not have any way to
zoom. CameraX does not appear to support zooming with any Glass gestures.
Therefore, the Glass client will only work if the camera is held close to the
Stirling engine.

## Open Workflow Version

We also created a version of this application that walks users through
assembling the engine, rather than disassembling it.
This runs using the code in the
[TwoStageOWF repository](https://github.com/cmusatyalab/TwoStageOWF).
You can download the accompanying files
[here](https://owncloud.cmusatyalab.org/owncloud/index.php/s/VLFP9UIPqYND2aX).
