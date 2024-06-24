import freefield
import numerosity_judgement
import spacial_unmasking
import localisation
import util

# Enter subject id here
sub_id = 101


# Initializing the setup
util.initialize_setup()


# Check for correct subject id
print(f"Beginning experiment with participant with subject id = {sub_id}.")


# Localisation accuracy
input("Press 'Enter' to start with localisation accuracy experiment.")

localisation.start_experiment(sub_id=sub_id, block_id=1, stim_type="pinknoise")
freefield.flush_buffers(processor="RX81")

input("Completed block 1/3. Press 'Enter' to continue with block 2 of localisation accuracy.")

localisation.start_experiment(sub_id=sub_id, block_id=2, stim_type="uso")
freefield.flush_buffers(processor="RX81")

input("Completed block 2/3. Press 'Enter' to continue with block 3 of localisation accuracy.")

localisation.start_experiment(sub_id=sub_id, block_id=3, stim_type="babble")
freefield.flush_buffers(processor="RX81")

input("Completed block 3/3. Ask participant questions of the questionnaire document. Press 'Enter' to continue.")


# Spatial unmasking
input("Press 'Enter"' to start with spatial unmasking experiment.')

spacial_unmasking.start_experiment(sub_id=sub_id, masker_type="babble", stim_type="syllable")

input("Spatial Unmasking finished. Ask remaining questionnaire questions...")


# Numerosity judgement
input("Press 'Enter' to start with numerosity judgement experiment.")

numerosity_judgement.start_experiment(sub_id=sub_id, block_id=1, stim_type="countries_forward", n_reps=10)

input("Completed block 1/2. Press 'Enter' to continue with block 2 of numerosity judgement.")

numerosity_judgement.start_experiment(sub_id=sub_id, block_id=2, stim_type="countries_reversed", n_reps=10)

input("Completed block 2/2. Ask participant questions of the questionnaire document. Press 'Enter' to continue.")


print("Experiment done.")


