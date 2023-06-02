from tqdm import tqdm
import pandas as pd
import numpy as np
import muspy
import glob
import os




meta_info = pd.read_csv("/data1/zhaojw/musicnet/musicnet_metadata.csv")
musicnet_midi_dir = '/data1/zhaojw/musicnet/musicnet_midis/'
meta_info = meta_info[meta_info.ensemble == 'String Quartet']

SPECIAL_FILES = {
    "1931_dv96_2": ["Violin 1", "Violin 2", "Viola", "Cello"] * 2,
    "1932_dv96_3": ["Violin 1", "Violin 2", "Viola", "Cello", "Empty", "Empty", "Empty", "Cello"],
    "1933_dv96_4": ["Violin 1", "Violin 2", "Viola", "Cello", "Empty", "Viola", "Empty", "Cello"],
    "2138_br51n1m2": ["Violin 1", "Violin 2"] * 2 + ["Viola"] + ["Cello"] * 3,
    "2140_br51n1m4": ["Violin 1", "Violin 2", "Viola", "Cello", "Cello"],
}


def get_instrument(name):
    """Return the instrument inferred from the track name."""
    if "viola" in name.lower():
        return "Viola"
    if "cello" in name.lower():
        return "Cello"
    for key in ("1st", "1st violin", "violin 1 arco", "violin 1", "violin1", "violino i", "violino i.", "violin1 sub", "violin1 pizz"):
        if key == name.lower():
            return "Violin 1"
    for key in ("2nd", "2nd violin", "violin 2 arco", "violin 2 pizz", "violin 2", "violin2", "violino ii", "violino ii.", "violin2 pizz"):
        if key == name.lower():
            return "Violin 2"
    return None



for idx in tqdm(meta_info.id):
    composer = meta_info[meta_info.id==idx].composer	

    midi_dir = glob.glob(os.path.join(musicnet_midi_dir, composer.values[0], f'{idx}*'))[0]

    music = muspy.read(midi_dir)
    music.adjust_resolution(4)

    song_len = 0
    for track in music.tracks:
        for note in track.notes:
            if note.time + note.duration > song_len:
                song_len = note.time + note.duration
    song_len = (song_len // 16 + 1) * 16

    quartet = {"Violin 1": np.zeros((song_len, 128)), \
                "Violin 2": np.zeros((song_len, 128)), \
                "Viola": np.zeros((song_len, 128)), \
                "Cello": np.zeros((song_len, 128))}

    if midi_dir.split('/')[-1].replace('.mid', '') in SPECIAL_FILES:
         instruments = SPECIAL_FILES[midi_dir.split('/')[-1].replace('.mid', '')]
         for i_tk, track in enumerate(music.tracks):
            if instruments[i_tk] == "Empty":
                continue
            for note in track.notes:
                if note.duration > 0:
                    quartet[instruments[i_tk]][note.time, note.pitch] = note.duration
    else:
        for track in music.tracks:
            if track.name.lower() in ['tempo']:
                continue
            instrument = get_instrument(track.name)
            for note in track.notes:
                if note.duration > 0:
                    quartet[instrument][note.time, note.pitch] = note.duration

        

    np.savez_compressed(f"/data1/zhaojw/musicnet/4_bin_quantization/{composer.values[0]}-{midi_dir.split('/')[-1].replace('.mid', '.npz')}", \
                                violin1 = quartet['Violin 1'], \
                                violin2 = quartet['Violin 2'], \
                                viola = quartet['Viola'], \
                                cello = quartet['Cello'])



    

    