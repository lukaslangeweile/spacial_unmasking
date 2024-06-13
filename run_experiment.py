import freefield

import numerosity_judgement
import spacial_unmasking
import localisation
import util

sub_id = 101
block_id = 99

"""util.initialize_setup()
localisation.start_experiment(sub_id, block_id, stim_type="pinknoise")
freefield.flush_buffers(processor="RX81")
localisation.start_experiment(sub_id, block_id, stim_type="uso")
freefield.flush_buffers(processor="RX81")
localisation.start_experiment(sub_id, block_id, stim_type="syllable")"""

util.initialize_setup()
spacial_unmasking.start_experiment(sub_id=sub_id, block_id=block_id, masker_type="pinknoise", stim_type="syllable")

"""util.initialize_setup()
input("Start")
numerosity_judgement.start_experiment(sub_id, block_id, "countries_reversed", 30)"""
