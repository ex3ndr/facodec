import torch
import torch.nn as nn
from .model_impl import FACodecEncoder, FACodecDecoder

class FACodec(nn.Module):
    def __init__(self):
        super(FACodec, self).__init__()
        self.encoder = FACodecEncoder(
            ngf = 32, 
            up_ratios = [2, 4, 5, 5], 
            out_channels = 256
        )
        self.decoder = FACodecDecoder(
            in_channels = 256,
            upsample_initial_channel = 1024,
            ngf = 32,
            up_ratios = [5, 5, 4, 2],
            vq_num_q_c = 2,
            vq_num_q_p = 1,
            vq_num_q_r = 3,
            vq_dim = 256,
            codebook_dim = 8,
            codebook_size_prosody = 10,
            codebook_size_content = 10,
            codebook_size_residual = 10,
            use_gr_x_timbre = True,
            use_gr_residual_f0 = True,
            use_gr_residual_phone = True,
        )

    @torch.no_grad()
    def speaker_embedding(self, source):
        assert source.dim() == 1, "Input tensor must be 1D"

        # Run Encoder
        enc_out = self.encoder(source.unsqueeze(0).unsqueeze(0))

        # Run Decoder
        return self.decoder.calculate_speaker_embedding(enc_out).squeeze(0)

    @torch.no_grad()
    def encode(self, source):
        assert source.dim() == 1, "Input tensor must be 1D"

        # Run Encoder
        enc_out = self.encoder(source.unsqueeze(0).unsqueeze(0))

        # Run Decoder
        vq_post_emb, vq_id, _, quantized, spk_embs = self.decoder(enc_out, eval_vq=False, vq=True)

        # Split codes
        prosody_code = vq_id[:1].squeeze(1)
        cotent_code = vq_id[1:3].squeeze(1)
        residual_code = vq_id[3:].squeeze(1)

        return prosody_code, cotent_code, residual_code, spk_embs.squeeze(0)

    @torch.no_grad()
    def decode(self, prosody_code, cotent_code, residual_code, spk_embs):

        # Merge codes
        if residual_code is None:    
            vq_id = torch.cat([prosody_code, cotent_code], dim=0).unsqueeze(1)
        else:
            vq_id = torch.cat([prosody_code, cotent_code, residual_code], dim=0).unsqueeze(1)

        # Run embedding
        vq_emb = self.decoder.vq2emb(vq_id, use_residual_code = residual_code is not None)

        # Run decoder
        return self.decoder.inference(vq_emb, spk_embs.unsqueeze(0)).squeeze(0).squeeze(0)

