import os
from torch import nn
from utils.training import kl_with_normal
import torch
from dl_modules import PtvaeEncoder, PianoTreeDecoder, FeatDecoder
import numpy as np
from utils.format_convert import grid2pr

from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from torch.distributions import Normal
import random


class PitchFunctionEncoder(nn.Module):
    """Function query-net for the pitch function"""
    def __init__(self, emb_size=256, z_dim=128, num_channel=10):
        super(PitchFunctionEncoder, self).__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, num_channel, kernel_size=12, stride=1, padding=0),
                                 nn.ReLU(),
                                 nn.MaxPool1d(kernel_size=4, stride=4))
        self.fc = nn.Linear(num_channel * 29, emb_size)
        self.linear_mu = nn.Linear(emb_size, z_dim)
        self.linear_var = nn.Linear(emb_size, z_dim)
        self.emb_size = emb_size
        self.z_dim = z_dim
        self.z2hid = nn.Linear(z_dim, emb_size)
        self.hid2out = nn.Linear(emb_size, 128)
        self.mse_func = nn.MSELoss()

    def forward(self, pr):
        # pr: (bs, 128)
        bs = pr.size(0)
        pr = pr.unsqueeze(1)
        pr = self.cnn(pr).reshape(bs, -1)
        pr = self.fc(pr)  # (bs, emb_size)

        mu = self.linear_mu(pr)
        var = self.linear_var(pr).exp_()

        dist = Normal(mu, var)
        return dist

    def decoder(self, z):
        return self.hid2out(torch.relu(self.z2hid(z)))

    def recon_loss(self, pred, func_gt):
        return self.mse_func(pred, func_gt)


class TimeFunctionEncoder(nn.Module):
    """Function query-net for the time function"""
    def __init__(self, emb_size=256, z_dim=128, num_channel=10):
        super(TimeFunctionEncoder, self).__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, num_channel, kernel_size=4, stride=4, padding=0),
                                 nn.ReLU())
        self.fc = nn.Linear(num_channel * 8, emb_size)

        self.linear_mu = nn.Linear(emb_size , z_dim)
        self.linear_var = nn.Linear(emb_size, z_dim)
        self.emb_size = emb_size
        self.z_dim = z_dim
        self.z2hid = nn.Linear(z_dim, emb_size)
        self.hid2out = nn.Linear(emb_size, 32)
        self.mse_func = nn.MSELoss()

    def forward(self, pr):
        # pr: (bs, 32)
        bs = pr.size(0)
        pr = pr.unsqueeze(1)
        pr = self.cnn(pr).reshape(bs, -1)
        pr = self.fc(pr)  # (bs, emb_size)

        mu = self.linear_mu(pr)
        var = self.linear_var(pr).exp_()

        dist = Normal(mu, var)
        return dist

    def decoder(self, z):
        return self.hid2out(torch.relu(self.z2hid(z)))

    def recon_loss(self, pred, func_gt):
        return self.mse_func(pred, func_gt)


class Query_and_reArrange(nn.Module):
    """Q&A model for multi-track rearrangement"""
    def __init__(self, name, device, trf_layers=2):
        super(Query_and_reArrange, self).__init__()

        self.name = name
        self.device = device
        
        # symbolic encoder
        self.prmat_enc_fltn = PtvaeEncoder(max_simu_note=32, device=self.device, z_size=256)

        # track function encoder
        self.func_pitch_enc = PitchFunctionEncoder(256, 128, 10)
        self.func_time_enc = TimeFunctionEncoder(256, 128, 10)

        # feat_dec + pianotree_dec = symbolic decoder
        self.feat_dec = FeatDecoder(z_dim=256)  # for symbolic feature recon
        self.feat_emb_layer = nn.Linear(3, 64)
        self.pianotree_dec = PianoTreeDecoder(z_size=256, feat_emb_dim=64, device=device)

        self.Transformer_layers = nn.ModuleDict({})
        self.trf_layers = trf_layers
        for idx in range(self.trf_layers):
            self.Transformer_layers[f'layer_{idx}'] = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=.1, activation=F.gelu, batch_first=True)

        self.prog_embedding = nn.Embedding(num_embeddings=35, embedding_dim=256, padding_idx=34)

        self.eq_feat_head = nn.Linear(256, 4)

        self.trf_mu = nn.Linear(256, 256)
        self.trf_var = nn.Linear(256, 256)

    
    def run(self, pno_tree_mix, prog, func_pitch, func_time, pno_tree=None, feat=None, track_pad_mask=None, tfr1=0, tfr2=0, inference=False, mel_id=None):
        """
        Forward path of the model in training (w/o computing loss).
        """

        batch, track, time = func_time.shape
        max_simu_note = 16
        #print('pno_tree', pno_tree.shape)
        dist_mix, _, _ = self.prmat_enc_fltn(pno_tree_mix)   #
        if inference:
            z_mix = dist_mix.mean
        else:
            z_mix = dist_mix.rsample()

        #print('pr_mat', pr_mat.shape)
        func_pitch = func_pitch.reshape(-1, 128)
        dist_fp = self.func_pitch_enc(func_pitch)

        func_time = func_time.reshape(-1, 32)
        dist_ft = self.func_time_enc(func_time)

        if inference:
            z_fp = dist_fp.mean
            z_ft = dist_ft.mean
        else:
            z_fp = dist_fp.rsample()
            z_ft = dist_ft.rsample()

        fp_recon = self.func_pitch_enc.decoder(z_fp).reshape(batch, track, -1)
        ft_recon = self.func_time_enc.decoder(z_ft).reshape(batch, track, -1)

        z_func = torch.cat([
                            z_fp.reshape(batch, track, -1),
                            z_ft.reshape(batch, track, -1)
                            ],
                            dim=-1) #(batch, track, 256),

        #print('prog', prog.shape)
        #print('prog embedding', self.prog_embedding(prog[:, 0]).shape)

        z = torch.cat([
                        z_mix.unsqueeze(1), #(batch, 1, 256)
                        z_func + self.prog_embedding(prog)],
                    dim=1)  #z: (batch, track+1, 256)"""

        if not inference:
            trf_mask = torch.cat([torch.zeros(batch, 1, device=z.device).bool(), track_pad_mask], dim=-1)   #(batch, track+1)
        else:
            trf_mask = torch.zeros(batch, track+1, device=z.device).bool()
        for idx in range(self.trf_layers):
            z = self.Transformer_layers[f'layer_{idx}'](src=z, src_key_padding_mask=trf_mask)

        # reconstruct symbolic feature using audio-texture repr.
        z = z[:, 1:].reshape(-1, 256)

        mu = self.trf_mu(z)
        var = self.trf_var(z).exp_()

        dist_trf = Normal(mu, var)
        if inference and (mel_id is None):
            z = dist_trf.mean
        elif inference and (mel_id is not None):
            z1 = dist_trf.mean.reshape(batch, track, 256)
            z2 = dist_trf.rsample().reshape(batch, track, 256)
            z = torch.cat([z1[:, :mel_id], z2[:, mel_id: mel_id+1], z1[:, mel_id+1:]], dim=1).reshape(-1, 256)
        else:
            z = dist_trf.rsample()
        #z = z.reshape(batch, track, 256)

        if not inference:
            feat = feat.reshape(-1, time, 3)
        recon_feat = self.feat_dec(z, inference, tfr1, feat)    #(batch*track, time, 3)
        # embed the reconstructed feature (without applying argmax)
        feat_emb = self.feat_emb_layer(recon_feat)

        # prepare the teacher-forcing data for pianotree decoder
        if inference:
            embedded_pno_tree = None
            pno_tree_lgths = None
        else:
            embedded_pno_tree, pno_tree_lgths = self.pianotree_dec.emb_x(pno_tree.reshape(-1, time, max_simu_note, 6))

        # pianotree decoder
        recon_pitch, recon_dur = \
            self.pianotree_dec(z, inference, embedded_pno_tree, pno_tree_lgths, tfr1, tfr2, feat_emb)

        recon_pitch = recon_pitch.reshape(batch, track, time, max_simu_note-1, 130)
        recon_dur = recon_dur.reshape(batch, track, time, max_simu_note-1, 5, 2)
        recon_feat = recon_feat.reshape(batch, track, time, 3)

        return recon_pitch, recon_dur, recon_feat, \
               fp_recon, ft_recon, \
               dist_mix, dist_fp, dist_ft, dist_trf

    def loss_function(self, pno_tree, feat, func_pitch, func_time, 
                      recon_pitch, recon_dur, recon_feat, fp_recon, ft_recon,
                      dist_mix, dist_fp, dist_ft, dist_trf, track_pad_mask,
                      beta_1, beta_2, weights):
        """ Compute the loss from ground truth and the output of self.run()"""
        # pianotree recon loss
        pno_tree_l, pitch_l, dur_l = \
            self.pianotree_dec.recon_loss(pno_tree[torch.logical_not(track_pad_mask)], 
                                          recon_pitch[torch.logical_not(track_pad_mask)], 
                                          recon_dur[torch.logical_not(track_pad_mask)],
                                          weights, False)
        # feature prediction loss
        feat_l, onset_feat_l, int_feat_l, center_feat_l = \
            self.feat_dec.recon_loss(feat[torch.logical_not(track_pad_mask)], recon_feat[torch.logical_not(track_pad_mask)])

        fp_l = self.func_pitch_enc.recon_loss(fp_recon, func_pitch)
        ft_l = self.func_time_enc.recon_loss(ft_recon, func_time)
        func_l = fp_l + ft_l

        # kl losses
        kl_mix = kl_with_normal(dist_mix)
        kl_fp = kl_with_normal(dist_fp)
        kl_ft = kl_with_normal(dist_ft)
        kl_trf = kl_with_normal(dist_trf)

        kl_l = beta_1 * (kl_mix + kl_trf) + beta_2 * (kl_fp + kl_ft)

        loss = pno_tree_l + feat_l + kl_l + func_l

        return loss, pno_tree_l, pitch_l, dur_l, \
               kl_l, kl_mix, kl_trf, kl_fp, kl_ft, \
               feat_l, onset_feat_l, int_feat_l, center_feat_l, \
               func_l, fp_l, ft_l

    def loss(self, pno_tree_mix, prog, func_pitch, func_time, pno_tree, feat, track_pad_mask, tfr1, tfr2,
             beta_1=0.01, beta_2=0.5, weights=(1, 0.5)):
        """forward and calculate loss"""
        output = self.run(pno_tree_mix, prog, func_pitch, func_time, pno_tree, feat, track_pad_mask, tfr1, tfr2)
        return self.loss_function(pno_tree, feat, func_pitch, func_time, *output, track_pad_mask, beta_1, beta_2, weights)
    
    def output_process(self, recon_pitch, recon_dur):
        grid_recon = torch.cat([recon_pitch.max(-1)[-1].unsqueeze(-1), recon_dur.max(-1)[-1]], dim=-1)
        _, track, _, max_simu_note, grid_dim = grid_recon.shape
        grid_recon = grid_recon.permute(1, 0, 2, 3, 4)
        grid_recon = grid_recon.reshape(track, -1, max_simu_note, grid_dim)
        pr_recon = np.array([grid2pr(matrix) for matrix in grid_recon.detach().cpu().numpy()])
        return pr_recon

    def inference(self, pno_tree_mix, prog, func_pitch, func_time, mel_id=None):
        self.eval()
        with torch.no_grad():
            recon_pitch, recon_dur, _, _, _, _, _, _, _ = self.run(pno_tree_mix, prog, func_pitch, func_time, inference=True, mel_id=mel_id)
            pr_recon = self.output_process(recon_pitch, recon_dur)
        return pr_recon

    def forward(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError

    
class Query_and_reArrange_vocie_separation(nn.Module):
    """ Q&A-V model for voice separation """
    def __init__(self, name, device, trf_layers=2):
        super(Query_and_reArrange_vocie_separation, self).__init__()

        self.name = name
        self.device = device
        
        # symbolic encoder
        self.prmat_enc_fltn = PtvaeEncoder(max_simu_note=32, device=self.device, z_size=256)

        # track function encoder
        self.func_pitch_enc = PitchFunctionEncoder(256, 128, 10)
        self.func_time_enc = TimeFunctionEncoder(256, 128, 10)

        # feat_dec + pianotree_dec = symbolic decoder
        self.feat_dec = FeatDecoder(z_dim=256)  # for symbolic feature recon
        self.feat_emb_layer = nn.Linear(3, 64)
        self.pianotree_dec = PianoTreeDecoder(z_size=256, feat_emb_dim=64, device=device)

        self.Transformer_layers = nn.ModuleDict({})
        self.trf_layers = trf_layers
        for idx in range(self.trf_layers):
            self.Transformer_layers[f'layer_{idx}'] = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=.1, activation=F.gelu, batch_first=True)

        self.prog_embedding = nn.Embedding(num_embeddings=35, embedding_dim=256, padding_idx=34)

        self.eq_feat_head = nn.Linear(256, 4)

        self.trf_mu = nn.Linear(256, 256)
        self.trf_var = nn.Linear(256, 256)

        #an additional GRU infering part (function in our case) from whole (mixture)
        hidden_dim_pw = 512
        z_mix_dim = 256
        z_mix_input_dim = 256
        output_dim_pw = 256
        self.z2dec_hid_pw = nn.Linear(z_mix_dim, hidden_dim_pw)
        self.z2dec_in_pw = nn.Linear(z_mix_dim, z_mix_input_dim)
        self.gru_pw = nn.GRU(output_dim_pw + z_mix_input_dim, hidden_dim_pw,
                          batch_first=True,
                          bidirectional=False)
        self.init_input_pw = nn.Parameter(torch.rand(output_dim_pw))
        self.out_pitch = nn.Linear(hidden_dim_pw//2, output_dim_pw//2)
        self.out_time = nn.Linear(hidden_dim_pw//2, output_dim_pw//2)

    def infer_track_function(self, z_mix, inference, tfr, func_pitch=None, func_time=None):
        #z_mix: (bs, 256)
        #func_pitch: (bs, 4, 128)
        #func_time: (bs, 4, 32)

        bs = z_mix.size(0)
        z_hid = self.z2dec_hid_pw(z_mix).unsqueeze(0)   #(1, bs, 512)
        z_in = self.z2dec_in_pw(z_mix).unsqueeze(1) #(bs, 1, 256)
        if inference:
            tfr = 0.
        token = self.init_input_pw.repeat(bs, 1).unsqueeze(1)   #(bs, 1, 256)
        
        out_fp = []
        out_ft = []

        repr_func = []

        for t in range(4):
            y_t, z_hid = \
                self.gru_pw(torch.cat([token, z_in], dim=-1), z_hid)
            fp_repr = self.out_pitch(y_t[:, :, :256])   #(bs, 1, 128)
            ft_repr = self.out_time(y_t[:, :, 256:])   #(bs, 1, 128)

            pred_fp = self.func_pitch_enc.decoder(fp_repr.squeeze(1)).unsqueeze(1)    #(bs, 1, 128)
            pred_ft = self.func_time_enc.decoder(ft_repr.squeeze(1)).unsqueeze(1)    #(bs, 1, 32)

            repr_func.append(torch.cat([fp_repr, ft_repr], dim=-1))
            out_fp.append(pred_fp)
            out_ft.append(pred_ft)

            if t == 4 - 1:
                break
            teacher_force = random.random() < tfr
            if teacher_force and not inference:
                token = torch.cat([
                                self.func_pitch_enc(func_pitch[:, t]).mean,
                                self.func_time_enc(func_time[:, t]).mean
                            ], dim=-1).unsqueeze(1) #(bs, 1, 256)
            else:
                token = torch.cat([
                                self.func_pitch_enc(pred_fp[:, 0]).mean,
                                self.func_time_enc(pred_ft[:, 0]).mean
                            ], dim=-1).unsqueeze(1) #(bs, 1, 256)
        
        recon_func_repr = torch.cat(repr_func, dim=1)   #(bs, 4, 256)
        recon_fp = torch.cat(out_fp, dim=1)   #(bs, 4, 128)
        recon_ft = torch.cat(out_ft, dim=1)   #(bs, 4, 32)
        return recon_func_repr, recon_fp, recon_ft

    def run(self, pno_tree_mix, prog, func_pitch, func_time, pno_tree=None, feat=None, track_pad_mask=None, tfr1=0, tfr2=0, inference=False, mel_id=None):
        """
        Forward path of the model in training (w/o computing loss).
        """
        #pno_tree: (batch, max_track, time, max_simu_note, 6)
        #chd: (batch, time', 36)
        #pr_fltn: (batch, max_track, time, 128)
        #prog: (batch, 5, max_track)
        #track_pad_mask: (batch, max_track)
        #feat: (batch, max_track, time, 3)
        #func_pitch: (batch, max_track, 128)
        #func_time: (batch, max_track, 32)

        _, track = prog.shape
        batch, time, _, _ = pno_tree_mix.shape
        max_simu_note = 16
        #print('pno_tree', pno_tree.shape)
        dist_mix, _, _ = self.prmat_enc_fltn(pno_tree_mix)   #
        if inference:
            z_mix = dist_mix.mean
        else:
            z_mix = dist_mix.rsample()
        
        z_func, fp_recon, ft_recon \
         = self.infer_track_function(z_mix, inference, tfr1, func_pitch, func_time)
        dist_fp = None
        dist_ft = None

        z = torch.cat([
                        z_mix.unsqueeze(1), #(batch, 1, 256)
                        z_func + self.prog_embedding(prog)],
                    dim=1)  #z: (batch, track+1, 256)"""


        if not inference:
            trf_mask = torch.cat([torch.zeros(batch, 1, device=z.device).bool(), track_pad_mask], dim=-1)   #(batch, track+1)
        else:
            trf_mask = torch.zeros(batch, track+1, device=z.device).bool()
        for idx in range(self.trf_layers):
            z = self.Transformer_layers[f'layer_{idx}'](src=z, src_key_padding_mask=trf_mask)

        # reconstruct symbolic feature using audio-texture repr.
        z = z[:, 1:].reshape(-1, 256)

        mu = self.trf_mu(z)
        var = self.trf_var(z).exp_()

        dist_trf = Normal(mu, var)
        if inference:
            z = dist_trf.mean
        else:
            z = dist_trf.rsample()
        #z = z.reshape(batch, track, 256)

        if not inference:
            feat = feat.reshape(-1, time, 3)
        recon_feat = self.feat_dec(z, inference, tfr1, feat)    #(batch*track, time, 3)
        # embed the reconstructed feature (without applying argmax)
        feat_emb = self.feat_emb_layer(recon_feat)

        # prepare the teacher-forcing data for pianotree decoder
        if inference:
            embedded_pno_tree = None
            pno_tree_lgths = None
        else:
            embedded_pno_tree, pno_tree_lgths = self.pianotree_dec.emb_x(pno_tree.reshape(-1, time, max_simu_note, 6))

        # pianotree decoder
        recon_pitch, recon_dur = \
            self.pianotree_dec(z, inference, embedded_pno_tree, pno_tree_lgths, tfr1, tfr2, feat_emb)

        recon_pitch = recon_pitch.reshape(batch, track, time, max_simu_note-1, 130)
        recon_dur = recon_dur.reshape(batch, track, time, max_simu_note-1, 5, 2)
        recon_feat = recon_feat.reshape(batch, track, time, 3)

        return recon_pitch, recon_dur, recon_feat, \
               fp_recon, ft_recon, \
               dist_mix, dist_fp, dist_ft, dist_trf

    def loss_function(self, pno_tree, feat, func_pitch, func_time, 
                      recon_pitch, recon_dur, recon_feat, fp_recon, ft_recon,
                      dist_mix, dist_fp, dist_ft, dist_trf, track_pad_mask,
                      beta_1, beta_2, weights):
        """ Compute the loss from ground truth and the output of self.run()"""
        # pianotree recon loss
        pno_tree_l, pitch_l, dur_l = \
            self.pianotree_dec.recon_loss(pno_tree[torch.logical_not(track_pad_mask)], 
                                          recon_pitch[torch.logical_not(track_pad_mask)], 
                                          recon_dur[torch.logical_not(track_pad_mask)],
                                          weights, False)

        # feature prediction loss
        feat_l, onset_feat_l, int_feat_l, center_feat_l = \
            self.feat_dec.recon_loss(feat[torch.logical_not(track_pad_mask)], recon_feat[torch.logical_not(track_pad_mask)])

        fp_l = self.func_pitch_enc.recon_loss(fp_recon, func_pitch)
        ft_l = self.func_time_enc.recon_loss(ft_recon, func_time)
        func_l = fp_l + ft_l

        # kl losses
        kl_mix = kl_with_normal(dist_mix)
        kl_fp = torch.tensor(0)#kl_with_normal(dist_fp)
        kl_ft = torch.tensor(0)#kl_with_normal(dist_ft)
        kl_trf = kl_with_normal(dist_trf)

        kl_l = beta_1 * (kl_mix + kl_trf) + beta_2 * (kl_fp + kl_ft)


        loss = pno_tree_l + feat_l + kl_l + func_l

        return loss, pno_tree_l, pitch_l, dur_l, \
               kl_l, kl_mix, kl_trf, kl_fp, kl_ft, \
               feat_l, onset_feat_l, int_feat_l, center_feat_l, \
               func_l, fp_l, ft_l

    def loss(self, pno_tree_mix, prog, func_pitch, func_time, pno_tree, feat, track_pad_mask, tfr1, tfr2,
             beta_1=0.01, beta_2=0.5, weights=(1, 0.5)):
        """forward and calculate loss"""
        output = self.run(pno_tree_mix, prog, func_pitch, func_time, pno_tree, feat, track_pad_mask, tfr1, tfr2)
        return self.loss_function(pno_tree, feat, func_pitch, func_time, *output, track_pad_mask, beta_1, beta_2, weights)
    
    def output_process(self, recon_pitch, recon_dur):
        grid_recon = torch.cat([recon_pitch.max(-1)[-1].unsqueeze(-1), recon_dur.max(-1)[-1]], dim=-1)
        _, track, _, max_simu_note, grid_dim = grid_recon.shape
        grid_recon = grid_recon.permute(1, 0, 2, 3, 4)
        grid_recon = grid_recon.reshape(track, -1, max_simu_note, grid_dim)
        pr_recon = np.array([grid2pr(matrix) for matrix in grid_recon.detach().cpu().numpy()])
        return pr_recon
    
    def pr2pr(self, pr):
        new_pr = np.zeros(pr.shape)
        for i_tk, track in enumerate(pr):
            for t, p in zip(*np.where(track > 0)):
                dur = track[t, p]
                #new_pr[i_tk, t: t+dur, p] = 1
                new_pr[i_tk, t, p] = 1
                new_pr[i_tk, min(t+dur, len(track))-1, p] = 1
        return new_pr
    
    def inference(self, pno_tree_mix, prog):
        self.eval()
        with torch.no_grad():
            recon_pitch, recon_dur, _, _, _, _, _, _, _ = self.run(pno_tree_mix, prog, None, None, inference=True)
            pr_recon = self.output_process(recon_pitch, recon_dur)
            pr_recon = self.pr2pr(pr_recon)

            #get nearest and 2nd-nearest neighbour
            pr_mix = grid2pr(pno_tree_mix.reshape(-1, 32, 6).detach().cpu().numpy(), max_note_count=32)
            time, pitch = np.nonzero(pr_mix)
            notes = np.stack((time, pitch), axis=-1)  #(n1, 2)
            note_wise_distance = []
            for track in pr_recon:
                time, pitch = np.nonzero(track)
                coordinates = np.stack((time, pitch), axis=-1)[np.newaxis, :, :]  #(1, n', 2)
                distance = np.min(np.sum(np.abs(notes[:, np.newaxis, :] - coordinates), axis=-1), axis=-1) #(n,)
                note_wise_distance.append(distance)
            note_wise_distance = np.array(note_wise_distance)  #(4, n)
            pred = np.argsort(note_wise_distance, axis=0)
            pred = pred[:2, :]
            distance = np.take_along_axis(note_wise_distance, pred, axis=0)

            # detect polyphonic voices
            time, pitch = np.nonzero(pr_mix)
            notes_se = np.stack((time, time+pr_mix[(time, pitch)]), axis=-1)   #(n, 2)
            s = notes_se[:, 0]
            e = notes_se[:, 1]
            relation = np.logical_not(
                                    np.logical_or(s[:, np.newaxis] - e[np.newaxis, :] >= 0,
                                                e[:, np.newaxis] - s[np.newaxis, :] <= 0)
                                                )   #(n, n)
            for i in range(len(relation)):
                relation[i, i] = False

            # in case of polyphonic voice, consider 2nd-nearest neighbour at lowest added distance
            for n1, n2 in zip(*np.nonzero(relation)):
                if pred[0, n1] == pred[0, n2]:
                    distance[0, n1] + distance[0, n2]
                    if distance[0, n1] + distance[1, n2] < distance[1, n1] + distance[0, n2]:
                        pred[:, n2] = pred[::-1, n2]
                        distance[:, n2] = distance[::-1, n2]
                    else:
                        pred[:, n1] = pred[::-1, n1]
                        distance[:, n1] = distance[::-1, n1]

            # reconstruct result
            reconstruction = np.zeros((4, pr_mix.shape[0], pr_mix.shape[1]))
            for idx, (t, p) in enumerate(zip(*np.nonzero(pr_mix))):
                reconstruction[pred[0, idx], t, p] = pr_mix[t, p]
            #assignment = np.sum((reconstruction > 0) * np.array([0, 1, 2, 3])[:, np.newaxis, np.newaxis], axis=0)
        return reconstruction

    def forward(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError
