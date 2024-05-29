import numerosity_judgement
import spacial_unmasking
import localisation

sub_id = 99

"""localisation.initialize()
localisation.start_trial(sub_id)"""

"""spacial_unmasking.initialize_setup()
spacial_unmasking.start_trial(sub_id=sub_id, masker_type="babble", stim_type="syllable")"""

numerosity_judgement.initialize_setup()
numerosity_judgement.start_trial(sub_id, n_reps=50)
