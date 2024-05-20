import numerosity_judgement
import spacial_unmasking
sub_id = 2
spacial_unmasking.initialize_setup()
spacial_unmasking.start_trial(sub_id=sub_id, masker_type="pinknoise", stim_type="syllable")
input("Press Enter to continue Experiment...")
spacial_unmasking.start_trial(sub_id=sub_id, masker_type="babble", stim_type="syllable")

"""numerosity_judgement.initialize_setup()
numerosity_judgement.start_trial(sub_id, n_reps=20)"""
