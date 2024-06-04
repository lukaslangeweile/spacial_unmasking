import numerosity_judgement
import spacial_unmasking
import localisation
import util

sub_id = 12

"""util.initialize_setup()
localisation.start_trial(sub_id, stim_type="pinknoise")"""

spacial_unmasking.initialize_setup()
spacial_unmasking.start_trial(sub_id=sub_id, masker_type="babble", stim_type="syllable")


"""numerosity_judgement.initialize_setup()
input("Start")
numerosity_judgement.start_trial(sub_id, n_reps=50)"""
