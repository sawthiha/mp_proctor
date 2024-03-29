# MediaPipe Proctoring Toolkit

[![DOI](https://zenodo.org/badge/592611238.svg)](https://zenodo.org/badge/latestdoi/592611238)

This repository presents face analytic algorithms for proctoring purposes based on [MediaPipe](https://github.com/google/mediapipe.git) framework.

## Features
- Eye Blink Detection
- Face Orientation (2 DoF, horizontal and vertical)
- Facial Activity
- Face Movement
- Facial embeddings
- Facial expression

## Requirement
- Mediapipe v0.8.10.2 (Simply checkout on [this commit](https://github.com/google/mediapipe/commit/63e679d9))

## Installation
To install the toolkit, you need to first install mediapipe and the checkout to the specific version, mentioned previously.
```bash
git clone -n https://github.com/google/mediapipe.git
cd mediapipe
git checkout 63e679d9
```
Then, you can clone this repository under mediapipe root directory.
```sh
git clone https://github.com/sawthiha/mp_proctor.git
```

## Demo Application
To run the demo app, you have to build it first:
```sh
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mp_proctor:demo_app
```
If there is no build error, you can run the application using the following command.
```sh
GLOG_logtostderr=1 bazel-bin/mp_proctor/demo_app \                     
  --calculator_graph_config_file=mp_proctor/graphs/proctor_cpu.pbtxt
```

## Troubleshooting

### Build errors
#### `fatal error: 'opencv2/core/version.hpp' file not found`
This error occurs when OpenCV installation or config is not detected. If you are on Linux, you can solve this by running the `setup_opencv.sh` provided by Mediapipe. You can find it in the root directory of MediaPipe.
```sh
chmod +x setup_opencv.sh
./setup_opencv.sh
```

## Related Research Papers
### Efficient Online Engagement Analytics Algorithm Toolkit That Can Run on Edge
Research Materials can be found on [this](https://github.com/sawthiha/r01_proctor_toolkit.git) repository.
```txt
@Article{a16020086,
AUTHOR = {Thiha, Saw and Rajasekera, Jay},
TITLE = {Efficient Online Engagement Analytics Algorithm Toolkit That Can Run on Edge},
JOURNAL = {Algorithms},
VOLUME = {16},
YEAR = {2023},
NUMBER = {2},
ARTICLE-NUMBER = {86},
URL = {https://www.mdpi.com/1999-4893/16/2/86},
ISSN = {1999-4893},
DOI = {10.3390/a16020086}
}
```