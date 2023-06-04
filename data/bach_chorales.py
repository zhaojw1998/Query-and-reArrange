import os
import shutil
import music21.corpus
import muspy
import numpy as np
from tqdm import tqdm


#This script is with credit to [Dong et al., 2021] https://github.com/salu133445/arranger/blob/main/arranger/data/collect_bach.py


# Get raw mxl and xml files of Bach chorales from music21
#for path in music21.corpus.getComposer("bach"):
#    if path.suffix in (".mxl", ".xml"):
#        shutil.copyfile(path, "./data/Bach_Chorales_mxl/" + path.name)


NAMES = {
    "soprano": "Soprano",
    "alto": "Alto",
    "tenor": "Tenor",
    "bass": "Bass",
}


SPECIAL_FILES = {
    "bwv171.6": {
        "Soprano\rOboe 1,2\rViolin1": "Soprano",
        "Alto\rViloin 2": "Alto",
        "Tenor\rViola": "Tenor",
        "Bass\rContinuo": "Bass",
    },
    "bwv41.6": {
        "Soprano Oboe 1 Violin1": "Soprano",
        "Alto Oboe 2 Viloin 2": "Alto",
        "Tenor Viola": "Tenor",
        "Bass Continuo": "Bass",
    },
    "bwv27.6": {
        "Soprano 1": "Soprano",
        "Soprano 2": "Soprano",
        "Alto": "Alto",
        "Tenor": "Tenor",
        "Bass": "Bass",
    },
    "bwv227.3": {
        "Soprano 1": "Soprano",
        "Soprano 2": "Soprano",
        "Alto": "Alto",
        "Tenor": "Tenor",
        "Bass": "Bass",
    }
}


for name in tqdm(os.listdir('./data/Bach_Chorales_mxl/')):
    m21 = music21.converter.parse(f'./data/Bach_Chorales_mxl/{name}')
    music = muspy.from_music21_score(m21)
    music.adjust_resolution(4)

    song_len = 0
    for track in music.tracks:
        for note in track.notes:
            if note.time + note.duration > song_len:
                song_len = note.time + note.duration
    song_len = (song_len // 16 + 1) * 16

    chorale = {"Soprano": np.zeros((song_len, 128)), \
                "Alto": np.zeros((song_len, 128)), \
                "Tenor": np.zeros((song_len, 128)), \
                "Bass": np.zeros((song_len, 128))}
    
    for track in music.tracks:
        if name[:-4] in SPECIAL_FILES:
            instrument = SPECIAL_FILES[str(name[:-4])].get(track.name)
        else:
            instrument = NAMES.get(track.name.lower())
        if instrument is None:
            continue
        for note in track.notes:
            chorale[instrument][note.time, note.pitch] = note.duration

    continue_flag = False
    for part in chorale:
        if len(np.nonzero(chorale[part])[1]) == 0:
            continue_flag = True
    if continue_flag:
        print(name)
        continue

    np.savez_compressed(f'./data/Bach_Chorales/{name[:-4]}.npz', \
                            soprano = chorale['Soprano'], \
                            alto = chorale['Alto'], \
                            tenor = chorale['Tenor'], \
                            bass = chorale['Bass'])
    
