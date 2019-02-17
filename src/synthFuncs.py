from synthesizer import Player, Synthesizer, Waveform
import time
import random

c = 16.35
csharp = 17.32
d = 18.35
dsharp = 19.45
e = 20.60
f = 21.83
fsharp = 23.12
g = 24.5
gsharp = 25.96
a = 27.5
asharp = 29.14
b = 30.87
# notes = ["c", "csharp", "d", "dsharp", "e", "f", "fsharp", "g", "gsharp", "a", "asharp", "b"]
notes = [c, csharp, d, dsharp, e, f, fsharp, g, gsharp, a, asharp, b]
key_offset = [0, 2, 4, 5, 7, 9, 11]
minor_key_offset = [0, 2, 3, 5, 7, 8, 10]
beats = [0.5, 1, 1.5, 2, 3, 4]


def note_octave(base_note, desired_octave):
    return base_note * pow(2, desired_octave)


def get_chord(root, chord_type):
    root_index = notes.index(root)
    if chord_type == 0:    # Major chord
        return [notes[root_index], notes[(root_index + 4) % 12], notes[(root_index + 7) % 12]]
    elif chord_type == 1:  # Minor chord
        return [notes[root_index], notes[(root_index + 3) % 12], notes[(root_index + 7) % 12]]
    elif chord_type == 2:  # Augmented Major chord
        return [notes[root_index], notes[(root_index + 4) % 12], notes[(root_index + 8) % 12]]
    else:                  # Diminished chord
        return [notes[root_index], notes[(root_index + 3) % 12], notes[(root_index + 6) % 12]]


def chord_octave(in_chord, octave, inversion):
    if len(in_chord) == 4:
        melody_note = in_chord[-1]
        chord = in_chord[:-1]
    else:
        chord = in_chord
    root_index = notes.index(chord[inversion])
    if root_index <= 3 and inversion == 2:
        octave += 1
    for i in range(len(chord)):
        if notes.index(chord[i]) < root_index:
            chord[i] = note_octave(chord[i], octave + 1)
        else:
            chord[i] = note_octave(chord[i], octave)
    if len(in_chord) == 4:
        in_chord[-1] = note_octave(melody_note, octave)

def get_progression_template():
    progression = random.randint(0, 4)
    if progression == 0:
        return [1, 4, 5, 1]
    elif progression == 1:
        return [1, 2, 5, 1]
    elif progression == 2:
        return [1, 6, 2, 5, 1]
    elif progression == 3:
        return [1, 3, 4, 2, 5, 1]
    elif progression == 4:
        return [1, 3, 6, 4, 2, 5, 1]


def get_RNchord(RNchord, major):
    majorList = [0, 1, 1, 0, 0, 1, 3]
    minorList = [1, 3, 0, 1, 1, 0, 0]
    if major:
        return majorList[RNchord-1]
    else:
        return minorList[RNchord-1]


"""
color      [0, 11]
    - 0  ->Red
    - 1  ->Orange
    - 2  ->Yellow 
    - 3  ->Yellow-Green 
    - 4  ->Green
    - 5  ->Cyan
    - 6  ->Light Blue
    - 7  ->Blue
    - 8  ->Dark Blue
    - 9  ->Purple
    - 10 ->Violet
    - 11 ->Pink
intensity  [0, 1]
    - 0  ->dark
    - 1  ->light
saturation [0, 4]
noise      [0, 24]
"""
def get_progression(color, saturation, intensity, noise):
    progression = get_progression_template()
    key_index = (color * 7) % 12
    major = intensity == 1
    octave = saturation + 1
    noise_index = noise * 4
    chord_progression = []
    for RNchord in progression:
        noise_quotient = random.randint(0, 99)
        if noise_quotient < noise_index:
            noise_out = 2
        else:
            noise_out = 0
        if major:
            chord = get_chord(notes[(key_index + key_offset[RNchord - 1]) % 12], get_RNchord(RNchord, major) + noise_out)
        else:
            chord = get_chord(notes[(((key_index + 9) % 12) + minor_key_offset[RNchord - 1]) % 12], get_RNchord(RNchord, major) + noise_out)
        chord_progression.append(chord)
    chord_progression = get_rythm(chord_progression)
    for measure in chord_progression:
        for chord in measure:
            chord_octave(chord[0], octave, random.randint(0, 2))
    return chord_progression


def play_progression(progression):
    player = Player()
    player.open_stream()
    synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=10.0, use_osc2=False)
    for measure in progression:
        for chord in measure:
            player.play_wave(synthesizer.generate_chord(chord[0], abs(chord[1])))


def generate_rythm():
    total = 4
    rythm_out = []
    while total != 0:
        beat_index = random.randint(0, 5)
        beat_in = beats[beat_index]
        if beat_index < 2:
            if random.randint(0, 2) == 0:
                beat_in *= -1
        if total - beats[beat_index] >= 0:
            rythm_out.append(beat_in)
            total -= beats[beat_index]
    return rythm_out


def get_rythm(progression):
    total_progression = []
    for i in range(len(progression)):
        rythm = generate_rythm()
        chord_notes = progression[i]
        progression_measure = []
        for beat in rythm:
            temp_progression = progression[i].copy()
            if beat > 0:
                note = chord_notes[random.randint(0, 2)]
                temp_progression.append(note)
            progression_measure.append((temp_progression, beat))
        total_progression.append(progression_measure)
    return total_progression


play_progression([])





