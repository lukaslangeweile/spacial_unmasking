import pathlib
import os
import slab

DIR = os.getcwd()
target_DIR = pathlib.Path(DIR + "/data/stim_files/pinknoise_burst_resamp_24414")

for i in range(10):
    noise = slab.Sound.pinknoise(duration=0.025, samplerate=24414, level=65)
    silence = slab.Sound.silence(duration=0.025, samplerate=24414)
    end_silence = slab.Sound.silence(duration=0.775, samplerate=24414)
    stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                               silence, noise, silence, noise, end_silence)
    stim = stim.ramp(when='both', duration=0.01)
    pinknoise = stim
    pinknoise.write(target_DIR / f"pinknoise_{i}.wav")