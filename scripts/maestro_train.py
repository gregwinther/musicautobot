from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.utils.midifile import *
from musicautobot.config import *
from musicautobot.music_transformer import *

import sys

sys.path.insert(0, "..")

midi_path = Path("data/midi/maestro_2018")
data_path = Path("data/numpy")
data_save_name = "maestro_init_data_save.pkl"


batch_size = 16
encode_position = True
dl_tfms = [batch_position_tfm] if encode_position else []
data = load_data(
    data_path,
    data_save_name,
    bs=batch_size,
    ecode_position=encode_position,
    dl_tfms=dl_tfms,
)

config = default_config()
config["encode_position"] = encode_position
learn = music_model_learner(data, config=config.copy())

import warnings

warnings.simplefilter("ignore", UserWarning)

learn.fit_one_cycle(4)

learn.save("maestro_2018_model")
