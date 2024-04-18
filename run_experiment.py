import spacial_unmasking
sub_id = 4

spacial_unmasking.initialize_setup()
spacial_unmasking.start_trial(sub_id=sub_id, masker_type="babble", stim_type="syllable")
input("Press Enter to continue Experiment...")
spacial_unmasking.start_trial(sub_id=sub_id, masker_type="pinknoise", stim_type="syllable")
