import freefield

import numerosity_judgement
import spacial_unmasking
import localisation
import util

sub_id = 101

"""util.initialize_setup()
localisation.start_trial(sub_id, stim_type="pinknoise")
freefield.flush_buffers(processor="RX81")
localisation.start_trial(sub_id, stim_type="uso")
freefield.flush_buffers(processor="RX81")
localisation.start_trial(sub_id, stim_type="syllable")"""

"""spacial_unmasking.initialize_setup()
spacial_unmasking.start_trial(sub_id=sub_id, masker_type="babble", stim_type="syllable")
"""

numerosity_judgement.initialize_setup()
input("Start")
numerosity_judgement.start_trial(sub_id, n_reps=30)
