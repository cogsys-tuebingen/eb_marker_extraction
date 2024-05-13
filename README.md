# Event-based marker extraction for [eWand](https://cogsys-tuebingen.github.io/ewand/) 

## Overview

This tool is built ontop of [FrequencyCam](https://github.com/berndpfrommer/frequency_cam). It relies on FrequencyCam's ability to detect frequencies. Additionally, it filters out frequencies outside of the frequency of eWand's markers, does blob detection and orders the markers.

## Installation

Please follow [the instructions of FrequencyCam](https://github.com/berndpfrommer/frequency_cam?tab=readme-ov-file#how-to-build).

## Usage

### Get trigger file

Use the `extract_triggers.py` scripts:

```
python3 extract_triggers.py /path/to/rosbag /path/to/trigger_file <optional arguments>
```

### Extract markers

1. Adjust the fields `bag_file`, `bag_topic` and `frame_time_file` in `eb_marker_extraction.launch.py`
2. Run colcon `colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo`
3. Run the extraction `os2 launch frequency_cam eb_marker_extraction.launch.py`

The extracted markers are then saved in `blob_detection_position_points.csv`.
