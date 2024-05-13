import argparse
from enum import Enum
from pathlib import Path
import numpy as np
from bag_reader_ros2 import BagReader
from event_camera_py import Decoder

class TriggerSource(Enum):
    FB_CAMERA = 1
    EXTERNAL = 2

def get_ext_trigger_timestamps(bag: Path):
    assert bag.exists()
    # assert rawfile.suffix == ".raw"

    topic = '/event_camera/events'
    bag_reader = BagReader(str(bag), topic)
    decoder = Decoder()

    all_triggers = []
    all_polarities = []

    while bag_reader.has_next():
        topic, msg, t_rec = bag_reader.read_next()
        decoder.decode(msg)
        triggers = decoder.get_ext_trig_events()

        if len(triggers) > 0:
            for trigger in triggers:
                all_triggers.append(trigger[1])
                all_polarities.append(trigger[0])

    time = np.array(all_triggers)
    pol = np.array(all_polarities)

    return time, pol

def get_reconstruction_timestamps(time: np.ndarray, pol: np.ndarray, trigger_source: TriggerSource, time_offset_us: int=0):
    assert 0 <= pol.max() <= 1
    assert np.all(np.abs(np.diff(pol)) == 1), 'polarity must alternate from trigger to trigger'

    timestamps = None
    if trigger_source == TriggerSource.FB_CAMERA:
        assert pol[0] == 1, 'first ext trigger polarity must be positive'
        assert pol[-1] == 0, 'last ext trigger polarity must be negative'
        # rising_ts = time[pol==1]
        falling_ts = time[pol==0]
        timestamps = falling_ts.astype('int64')
    else:
        timestamps = time[pol==1]
        timestamps = timestamps + time_offset_us

    return timestamps * 1e3

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read trigger data from prophesee raw file')
    parser.add_argument('bag')
    parser.add_argument('output_file', help='Path to output text file with timestamps for reconstruction.')
    parser.add_argument('--fb-source', '-fbs', action='store_true',
            help='The frame-based camera is the trigger source.')
    parser.add_argument('--external-source', '-es', action='store_true',
            help='An external trigger source is used.')
    parser.add_argument('--time_offset_us', '-offset', type=int, default=0,
            help='Add a constant time offset to the timestamp for reconstruction (Only for external source).'
            )

    args = parser.parse_args()

    bag = Path(args.bag)
    outfile = Path(args.output_file)
    assert not outfile.exists()
    assert outfile.suffix == '.txt'

    trigger_source = TriggerSource.FB_CAMERA
    if args.external_source:
        trigger_source = TriggerSource.EXTERNAL

    times, polarities = get_ext_trigger_timestamps(bag)
    reconstruction_ts = get_reconstruction_timestamps(times, polarities, trigger_source, args.time_offset_us)
    np.savetxt(str(outfile), reconstruction_ts, '%i')
