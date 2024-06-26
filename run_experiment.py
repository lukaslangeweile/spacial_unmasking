import freefield
import numerosity_judgement
import spacial_unmasking
import localisation
import util

# Enter subject id here
sub_id = input("Enter sub_id here...")
sub_id = "sub_" + sub_id
# Initializing the setup
util.initialize_setup()


# Check for correct subject id
print(f"Beginning experiment with participant with subject id = {sub_id}.")

"""# Spatial unmasking
input("Press 'Enter"' to start with spatial unmasking experiment.')

spacial_unmasking.start_experiment(sub_id=sub_id, masker_type="babble", stim_type="syllable")

input("Spatial Unmasking finished. Ask remaining questionnaire questions...")

spacial_unmasking.plot_target_ratio_vs_distance(sub_id=sub_id, masker_type="babble")



# Numerosity judgement
input("Press 'Enter' to start with numerosity judgement experiment.")

numerosity_judgement.start_experiment(sub_id=sub_id, block_id=1, stim_type="countries_forward", n_reps=10)"""

input("Completed block 1/4. Press 'Enter' to continue with block 2 of numerosity judgement.")

numerosity_judgement.start_experiment(sub_id=sub_id, block_id=2, stim_type="countries_forward", n_reps=10)

input("Completed block 2/4. Press 'Enter' to continue with block 3 of numerosity judgement.")

numerosity_judgement.start_experiment(sub_id=sub_id, block_id=3, stim_type="countries_reversed", n_reps=10)

input("Completed block 3/4. Press 'Enter' to continue with block 3 of numerosity judgement.")

numerosity_judgement.start_experiment(sub_id=sub_id, block_id=4, stim_type="countries_reversed", n_reps=10)

input("Completed block 4/4. Ask participant questions of the questionnaire document. Press 'Enter' to continue.")

numerosity_judgement.plot_results(sub_id=sub_id)

print("Experiment done.")


# Localisation accuracy
input("Press 'Enter' to start with localisation accuracy experiment.")

localisation.start_experiment(sub_id=sub_id, block_id=1, stim_type="pinknoise", n_reps=5)
freefield.flush_buffers(processor="RX81")

input("Completed block 1/4. Press 'Enter' to continue with block 2 of localisation accuracy.")

localisation.start_experiment(sub_id=sub_id, block_id=2, stim_type="pinknoise", n_reps=5)
freefield.flush_buffers(processor="RX81")

input("Completed block 2/4. Press 'Enter' to continue with block 3 of localisation accuracy.")

localisation.start_experiment(sub_id=sub_id, block_id=3, stim_type="babble", n_reps=5)
freefield.flush_buffers(processor="RX81")

input("Completed block 3/4. Press 'Enter' to continue with block 4 of localisation accuracy.")

localisation.start_experiment(sub_id=sub_id, block_id=4, stim_type="babble", n_reps=5)
freefield.flush_buffers(processor="RX81")

input("Completed block 4/4. Ask participant questions of the questionnaire document. Press 'Enter' to continue.")

localisation.plot_results(sub_id=sub_id)