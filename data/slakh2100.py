import os
import numpy as np
import pretty_midi as pyd
from tqdm import tqdm
from scipy.interpolate import interp1d
import yaml


def midi2matrix(midi, quaver):
        """
        Convert multi-track midi to a 3D matrix of shape (Track, Time, 128). 
        Each cell is a integer number representing quantized duration.
        """
        pr_matrices = []
        programs = []
        quantization_error = []
        for track in midi.instruments:
            qt_error = [] # record quantization error
            pr_matrix = np.zeros((len(quaver), 128, 2))
            for note in track.notes:
                note_start = np.argmin(np.abs(quaver - note.start))
                note_end =  np.argmin(np.abs(quaver - note.end))
                if note_end == note_start:
                    note_end = min(note_start + 1, len(quaver) - 1) # guitar/bass plunk typically results in a very short note duration. These note should be quantized to 1 instead of 0.
                pr_matrix[note_start, note.pitch, 0] = note_end - note_start
                pr_matrix[note_start, note.pitch, 1] = note.velocity

                #compute quantization error. A song with very high error (e.g., triple-quaver songs) will be discriminated and therefore discarded.
                if note_end == note_start:
                    qt_error.append(np.abs(quaver[note_start] - note.start) / (quaver[note_start] - quaver[note_start-1]))
                else:
                    qt_error.append(np.abs(quaver[note_start] - note.start) / (quaver[note_end] - quaver[note_start]))
                
            control_matrix = np.ones((len(quaver), 128, 1)) * -1
            for control in track.control_changes:
                #if control.time < time_end:
                #    if len(quaver) == 0:
                #        continue
                control_time = np.argmin(np.abs(quaver - control.time))
                control_matrix[control_time, control.number, 0] = control.value

            pr_matrix = np.concatenate((pr_matrix, control_matrix), axis=-1)
            pr_matrices.append(pr_matrix)
            programs.append(track.program)
            quantization_error.append(np.mean(qt_error))

        return np.array(pr_matrices), programs, quantization_error

ACC = 4 # quantize each beat as 4 positions


# Get raw MIDI of Slakh2100 from https://zenodo.org/record/4599666.
slakh_root = './data/slakh2100_flac_redux/'
save_root = './data/Slakh2100'
for split in ['train', 'validation', 'test']:
    slakh_split = os.path.join(slakh_root, split)
    save_split = os.path.join(save_root, split)
    if not os.path.exists(save_split):
        os.makedirs(save_split)
    print(f'processing {split} set ...')
    for song in tqdm(os.listdir(slakh_split)):
        break_flag = 0

        all_src_midi = pyd.PrettyMIDI(os.path.join(slakh_split, song, 'all_src.mid'))
        for ts in all_src_midi.time_signature_changes:
            if not (((ts.numerator == 2) or (ts.numerator == 4)) and (ts.denominator == 4)):
                break_flag = 1
                break
        if break_flag:
            continue    # process only 2/4 and 4/4 songs

        tracks = os.path.join(slakh_split, song, 'MIDI')
        track_names = os.listdir(tracks)
        track_midi = [pyd.PrettyMIDI(os.path.join(tracks, track)) for track in track_names]
        track_meta = yaml.safe_load(open(os.path.join(slakh_split, song, 'metadata.yaml'), 'r'))['stems']

        if len(all_src_midi.get_beats()) >= max([len(midi.get_beats()) for midi in track_midi]):
            beats = all_src_midi.get_beats()
            downbeats = all_src_midi.get_downbeats()
        else:
            beats = track_midi[np.argmax([len(midi.get_beats()) for midi in track_midi])].get_beats()
            downbeats = track_midi[np.argmax([len(midi.get_beats()) for midi in track_midi])].get_downbeats()

        beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
        quantize = interp1d(np.array(range(0, len(beats))) * ACC, beats, kind='linear')
        quaver = quantize(np.array(range(0, (len(beats) - 1) * ACC)))
        
        pr_matrices = []
        programs = []
        dynamic_matrices = []
        
        break_flag = 0
        for idx, midi in enumerate(track_midi):
            meta = track_meta[track_names[idx].replace('.mid', '')]
            if meta['is_drum']:
                continue    #let's skip drum for now
            else:
                pr_matrix, _, track_qt = midi2matrix(midi, quaver)
                if track_qt[0] > .2:
                    break_flag = 1
                    break
                pr_matrices.append(pr_matrix[..., 0])
                dynamic_matrices.append(pr_matrix[..., 1:])
                programs.append(meta['program_num'])
        if break_flag:
            continue    #skip the pieces with very large quantization error. This pieces are possibly triple-quaver songs
        
        pr_matrices = np.concatenate(pr_matrices, axis=0)
        programs = np.array(programs)
        dynamic_matrices = np.concatenate(dynamic_matrices, axis=0)

        downbeat_indicator = np.array([int(t in downbeats) for t in quaver])

        np.savez_compressed(os.path.join(save_split, f'{song}.npz'),\
                    tracks = pr_matrices,\
                    programs = programs,\
                    db_indicator = downbeat_indicator,\
                    dynamics = dynamic_matrices)
        
        #break



        