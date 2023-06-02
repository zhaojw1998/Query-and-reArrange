import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import random


ACC = 16            #quantize every 4 beats into 16 positions
SAMPLE_LEN = 32     #each sample has 2 bars
BAR_HOP_LEN = 1     #hop size is 1 bar
AUG_P = np.array([2, 2, 5, 5, 3, 7, 7, 5, 7, 3, 5, 1])  #prior for pitch transposition augmentation
NUM_INSTR_CLASS = 34    #number of supported instruments

#Supported instrument programs in Slakh2100 dataset
SLAKH_CLASS_PROGRAMS = dict({
    0: 'Acoustic Piano',    #0
    4: 'Electric Piano',    #1
    8: 'Chromatic Percussion',#2
    16: 'Organ',    #3
    24: 'Acoustic Guitar',  #4
    26: 'Clean Electric Guitar',    #5
    29: 'Distorted Electric Guitar',    #6
    32: 'Acoustic Bass',    #7
    33: 'Electric Bass',    #8
    40: 'Violin',   #9
    41: 'Viola',    #10
    42: 'Cello',    #11
    43: 'Contrabass',   #12
    46: 'Orchestral Harp',  #13
    47: 'Timpani',  #14
    48: 'String Ensemble',  #15
    50: 'Synth Strings',    #16
    52: 'Choir and Voice',  #17
    55: 'Orchestral Hit',   #18
    56: 'Trumpet',  #19
    57: 'Trombone', #20
    58: 'Tuba', #21
    60: 'French Horn',  #22
    61: 'Brass Section',    #23
    64: 'Soprano/Alto Sax', #24
    66: 'Tenor Sax',    #25
    67: 'Baritone Sax', #26
    68: 'Oboe', #27
    69: 'English Horn', #28
    70: 'Bassoon',  #29
    71: 'Clarinet', #30
    72: 'Pipe', #31
    80: 'Synth Lead',   #32
    88: 'Synth Pad' #33
})

#map an arbituary program to a supported Slakh2100 program
SLAKH_PROGRAM_MAPPING = dict({0: 0, 1: 0, 2: 0, 3: 0, 4: 4, 5: 4, 6: 4, 7: 4,\
                            8: 8, 9: 8, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8,\
                            16: 16, 17: 16, 18: 16, 19: 16, 20: 16, 21: 16, 22: 16, 23: 16,\
                            24: 24, 25: 24, 26: 26, 27: 26, 28: 26, 29: 29, 30: 29, 31: 29,\
                            32: 32, 33: 33, 34: 33, 35: 33, 36: 33, 37: 33, 38: 33, 39: 33,\
                            40: 40, 41: 41, 42: 42, 43: 43, 44: 43, 45: 43, 46: 46, 47: 47,\
                            48: 48, 49: 48, 50: 50, 51: 50, 52: 52, 53: 52, 54: 52, 55: 55,\
                            56: 56, 57: 57, 58: 58, 59: 58, 60: 60, 61: 61, 62: 61, 63: 61,\
                            64: 64, 65: 64, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70, 71: 71,\
                            72: 72, 73: 72, 74: 72, 75: 72, 76: 72, 77: 72, 78: 72, 79: 72,\
                            80: 80, 81: 80, 82: 80, 83: 80, 84: 80, 85: 80, 86: 80, 87: 80,\
                            88: 88, 89: 88, 90: 88, 91: 88, 92: 88, 93: 88, 94: 88, 95: 88})

#map Slakh2100 programs to continuous indices for embedding purposes
EMBED_PROGRAM_MAPPING = dict({
    0: 0, 4: 1, 8: 2, 16: 3, 24: 4, 26: 5, 29: 6, 32: 7,\
    33: 8, 40: 9, 41: 10, 42: 11, 43: 12, 46: 13, 47: 14, 48: 15,\
    50: 16, 52: 17, 55: 18, 56: 19, 57: 20, 58: 21, 60: 22, 61: 23, 
    64: 24, 66: 25, 67: 26, 68: 27, 69: 28, 70: 29, 71: 30, 72: 31,\
    80: 32, 88: 33})


class Slakh2100_Pop909_Dataset(Dataset):
    def __init__(self, slakh_dir, pop909_dir, sample_len=SAMPLE_LEN, hop_len=BAR_HOP_LEN, debug_mode=False, split='train', mode='train', with_dynamics=False, with_drums=False, merge_pop909=0):
        super(Slakh2100_Pop909_Dataset, self).__init__()
        self.split = split
        self.mode = mode
        self.debug_mode = debug_mode

        self.with_dynamics = with_dynamics
        self.with_drums = with_drums
        self.merge_pop909 = merge_pop909

        self.memory = dict({'tracks': [],
                            'programs': [],
                            'dynamics': [],
                            'drums': [],
                            'drum_programs': [],
                            'dir': []
                            })
        self.anchor_list = []
        self.sample_len = sample_len
        
        if slakh_dir is not None:
            print('loading Slakh2100 Dataset ...')
            self.load_data(slakh_dir, sample_len, hop_len)
        if pop909_dir is not None:
            print('loading Pop909 Dataset ...')
            self.load_data(pop909_dir, sample_len, hop_len)

    def __len__(self):
        return len(self.anchor_list)
    
    def __getitem__(self, idx):
        song_id, start = self.anchor_list[idx]
        tracks_sample = self.memory['tracks'][song_id][:, start: start+self.sample_len]
        program_sample = self.memory['programs'][song_id]
        if self.mode == 'train':    #delete empty tracks if any
            non_empty = np.nonzero(np.sum(tracks_sample, axis=(1, 2)))[0]
            tracks_sample = tracks_sample[non_empty]
            program_sample = program_sample[non_empty]
        if ((len(program_sample) <= 3) and (program_sample == 0).all()):
            #merge pop909 into a single piano track at certain probability
            if np.random.rand() < self.merge_pop909:    
                tracks_sample = np.max(tracks_sample, axis=0, keepdims=True)
                program_sample = np.array([0])

        if self.with_dynamics:
            dynamics = self.memory['dynamics'][song_id][:, start: start+self.sample_len]
        else: 
            dynamics = None
        if self.with_drums:
            drums = self.memory['drums'][song_id][:, start*4: (start+self.sample_len)*4]
            drum_programs = self.memory['drum_programs'][song_id]
        else:
            drums, drum_programs = None, None
        
        return tracks_sample, program_sample, (dynamics, drums, drum_programs), self.memory['dir'][song_id]


    def slakh_program_mapping(self, programs):
        return np.array([EMBED_PROGRAM_MAPPING[SLAKH_PROGRAM_MAPPING[program]] for program in programs])


    def load_data(self, data_dir, sample_len, hop_len):
        song_list = [os.path.join(data_dir, self.split, item) for item in os.listdir(os.path.join(data_dir, self.split))]
        if self.debug_mode:
            song_list = song_list[: 10]
        for song_dir in tqdm(song_list):
            song_data = np.load(song_dir)
            tracks = song_data['tracks']   #(n_track, time, 128)
            if 'programs' in song_data:
                programs = song_data['programs']    #(n_track, )
            else:
                programs = np.array([0]*len(tracks))

            """clipping""" 
            if (self.mode == 'train') and (self.split =='validation'):
                # during model training, no overlapping for validation set
                for i in range(0, tracks.shape[1], sample_len):
                    if i + sample_len >= tracks.shape[1]:
                        break
                    self.anchor_list.append((len(self.memory['tracks']), i))  #(song_id, start, total_length)
            else:
                # otherwise, hop size is 1-bar
                downbeats = np.nonzero(song_data['db_indicator'])[0]
                for i in range(0, len(downbeats), hop_len):
                    if downbeats[i] + sample_len >= tracks.shape[1]:
                        break
                    self.anchor_list.append((len(self.memory['tracks']), downbeats[i]))  #(song_id, start)
                #self.anchor_list.append((len(self.memory['tracks']), max(0, (tracks.shape[1]-sample_len))))

            self.memory['tracks'].append(tracks)
            self.memory['programs'].append(self.slakh_program_mapping(programs))
            self.memory['dir'].append(song_dir)

            if self.with_dynamics:
                self.memory['dynamics'].append(song_data['dynamics'])
            if self.with_drums:
                self.memory['drums'].append(song_data['drums'])
                self.memory['drum_programs'].append(song_data['drum_programs'])


def collate_fn(batch, device, pitch_shift=True):
    max_tracks = max([len(item[0]) for item in batch])

    tracks = [] 
    mixture = []
    instrument = []
    aux_feature = []
    mask = []   #track-wise pad mask
    function_pitch = []
    function_time = []

    if pitch_shift:
        aug_p = AUG_P / AUG_P.sum()
        aug_shift = np.random.choice(np.arange(-6, 6), 1, p=aug_p)[0]
    else:
        aug_shift = 0

    for pr, programs, (_, _, _), _ in batch:
        pr = pr_mat_pitch_shift(pr, aug_shift)
        aux, fp, ft = compute_pr_feat(pr)
        mask.append([0]*len(pr) + [1]*(max_tracks-len(pr)))

        pr = np.pad(pr, ((0, max_tracks-len(pr)), (0, 0), (0, 0)), mode='constant', constant_values=(0,))
        programs = np.pad(programs, (0, max_tracks-len(programs)), mode='constant', constant_values=(NUM_INSTR_CLASS,))
        aux = np.pad(aux, ((0, max_tracks-len(aux)), (0, 0), (0, 0)), mode='constant', constant_values=(0,))
        fp = np.pad(fp, ((0, max_tracks-len(fp)), (0, 0)), mode='constant', constant_values=(0,))
        ft = np.pad(ft, ((0, max_tracks-len(ft)), (0, 0)), mode='constant', constant_values=(0,))

        mix = pr2grid(np.max(pr, axis=0), max_note_count=32)
        grid = np.array([pr2grid(matrix) for matrix in pr])

        tracks.append(grid)
        mixture.append(mix)
        instrument.append(programs)
        aux_feature.append(aux)
        function_pitch.append(fp)
        function_time.append(ft)

    return  torch.from_numpy(np.array(mixture)).long().to(device), \
            torch.from_numpy(np.array(instrument)).to(device), \
            torch.from_numpy(np.array(function_pitch)).float().to(device), \
            torch.from_numpy(np.array(function_time)).float().to(device),\
            torch.from_numpy(np.array(tracks)).long().to(device), \
            torch.from_numpy(np.array(aux_feature)).float().to(device), \
            torch.BoolTensor(mask).to(device)


def collate_fn_inference(batch, device):
    assert len(batch) == 1
    tracks, instrument, (dynamics, drums, drum_programs), song_dir = batch[0]

    track, time, _ = tracks.shape
    if time % 32 != 0:
        pad_len = (time//32+1)*32 - time
        tracks = np.pad(tracks, ((0, 0), (0, pad_len), (0, 0)))
        if dynamics is not None:
            dynamics = np.pad(dynamics, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            dynamics[:, -pad_len:, :, -1] = -1
        if drums is not None:
            drums = np.pad(drums, ((0, 0), (0, pad_len*4), (0, 0), (0, 0)))
            drums[:, -pad_len*4:, :, -1] = -1
    tracks = tracks.reshape(track, -1, 32, 128).transpose(1, 0, 2, 3)

    _, function_pitch, function_time = compute_pr_feat(tracks)

    mixture = np.array([pr2grid(matrix, max_note_count=32) for matrix in np.max(tracks, axis=1)])

    mixture = torch.from_numpy(mixture).long().to(device)
    instrument = torch.from_numpy(instrument).repeat(tracks.shape[0], 1).to(device)
    function_pitch = torch.from_numpy(np.array(function_pitch)).float().to(device)
    function_time = torch.from_numpy(np.array(function_time)).float().to(device)

    return (mixture, instrument, function_pitch, function_time), (dynamics, drums, drum_programs), song_dir
    
        
def pr_mat_pitch_shift(pr_mat, shift):
    pr_mat = pr_mat.copy()
    pr_mat = np.roll(pr_mat, shift, -1)
    return pr_mat


def pr2grid(pr_mat, max_note_count=16, max_pitch=127, min_pitch=0,
                       pitch_pad_ind=130, dur_pad_ind=2,
                       pitch_sos_ind=128, pitch_eos_ind=129):
    """pr_mat: (32, 128)"""
    sample_len = len(pr_mat)
    grid = np.ones((sample_len, max_note_count, 6), dtype=int) * dur_pad_ind
    grid[:, :, 0] = pitch_pad_ind
    grid[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(sample_len, dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        if cur_idx[t] == max_note_count - 1:
            continue
        grid[t, cur_idx[t], 0] = p - min_pitch
        binary = np.binary_repr(min(int(pr_mat[t, p]), 32) - 1, width=5)
        grid[t, cur_idx[t], 1: 6] = \
            np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        cur_idx[t] += 1
    grid[np.arange(0, sample_len), cur_idx, 0] = pitch_eos_ind
    return grid


def compute_pr_feat(pr):
    #pr: (track, time, 128)
    onset = (np.sum(pr, axis=-1) > 0) * 1.   #(track, time)
    rhy_intensity = np.clip(np.sum((pr > 0) * 1., axis=-1) / 14, a_min=None, a_max=1)    #(track, time)

    weight = np.sum(pr, axis=-1)
    weight[weight==0] = 1
    pitch_center = np.sum(np.arange(0, 128)[np.newaxis, np.newaxis, :] * pr, axis=-1) / weight / 128

    feature = np.stack((onset, rhy_intensity, pitch_center), axis=-1)

    func_pitch = np.sum((pr > 0) * 1., axis=-2) / 32

    func_time = rhy_intensity.copy()
    
    return feature, func_pitch, func_time


BACH_CHORALES_PROGRAMS = dict({
    52: 'Saprano',  #0
    52: 'Alto',     #1
    52: 'Tenor',    #2
    52: 'Bass',     #3
})


STRING_QUARTETS_PROGRAMS = dict({
    40: 'violin1',
    40: 'violin2',
    41: 'viola',
    42: 'cello',
})


class Voice_Separation_Dataset(Dataset):
    def __init__(self, bach_dir, quartets_dir, sample_len=SAMPLE_LEN, hop_len=BAR_HOP_LEN, debug_mode=False, split='train', mode='train', fold=0):
        super(Voice_Separation_Dataset, self).__init__()
        self.split = split
        self.mode = mode
        self.fold=fold
        self.debug_mode = debug_mode

        self.memory = dict({'voices': [],
                            'dir': []
                            })
        self.anchor_list = []
        self.sample_len = sample_len

        if bach_dir is not None:
            assert quartets_dir is None
            self.programs = np.array([0, 1, 2, 3])
            print('loading Bach Chorale Dataset ...')
            self.load_data(bach_dir, sample_len, hop_len)

        elif quartets_dir is not None:
            assert bach_dir is None
            self.programs = np.array([9, 9, 10, 11])
            print('loading String Quartets Dataset ...')
            self.load_data(quartets_dir, sample_len, hop_len)
        

    def __len__(self):
        return len(self.anchor_list)
    
    def __getitem__(self, idx):
        song_id, start = self.anchor_list[idx]
        if type(self.sample_len) == int:
            vocies_sample = self.memory['voices'][song_id][:, start: start+self.sample_len]
        elif self.sample_len == 'full':
            vocies_sample = self.memory['voices'][song_id][:, start:]
        program_sample = self.programs

        return vocies_sample, program_sample, (None, None, None), self.memory['dir'][song_id]


    def load_data(self, data_dir, sample_len, hop_len):
        data_list = os.listdir(data_dir)
        random.seed(0)
        random.shuffle(data_list)
        folds = {}
        for i in range(10):
            folds[i] = data_list[int(len((data_list))*i/10): int(len(data_list)*(i+1)/10)]
        if self.split == 'train':
            data_list = []
            for i in range(2, 10):
                data_list += folds[(self.fold+i)%10]
        elif self.split == 'validation':
            data_list = folds[(self.fold+1)%10]
        elif self.split == 'test':
            data_list = folds[self.fold]
        elif self.split == 'full':
            data_list = []
            for i in range(0, 10):
                data_list += folds[i]
        if self.debug_mode:
            data_list = data_list[: 10]
        for song_dir in tqdm(data_list):
            song = np.load(os.path.join(data_dir, song_dir))
            try:
                song = np.stack([song['soprano'], song['alto'], song['tenor'], song['bass']], axis=0)    #(4, time, 128)
            except KeyError:
                song = np.stack([song['violin1'], song['violin2'], song['viola'], song['cello']], axis=0)    #(4, time, 128)

            if type(sample_len) == int:
                for idx in range(0, song.shape[1], hop_len*16):
                    if idx + sample_len > song.shape[1]:
                        break
                    self.anchor_list.append((len(self.memory['voices']), idx))
            elif sample_len == 'full':
                self.anchor_list.append((len(self.memory['voices']), 0))

            self.memory['voices'].append(song)
            self.memory['dir'].append(song_dir)

        #record = []
        #for key in folds:
        #    record.append(f'fold {key}\n')
        #    for name in folds[key]:
        #        record.append(f'{name}\n')
        #    record.append('\n')#
        #with open(f'./fold_record.txt', 'w') as f:
        #    f.writelines(record)