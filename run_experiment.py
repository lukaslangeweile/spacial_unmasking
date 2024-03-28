import spacial_unmasking
sub_id=100

spacial_unmasking.initialize_setup()
spacial_unmasking.start_trial(sub_id=sub_id, masker_type="babble", stim_type="syllable", normalisation_method="rms_pinknoise")
input("Press Enter to continue Experiment...")
spacial_unmasking.start_trial(sub_id=sub_id, masker_type="pink_noise", stim_type="syllable", normalisation_method="rms_pinknoise")
