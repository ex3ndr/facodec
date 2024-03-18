dependencies = ['torch', 'torchaudio']

def facodec(pretrained=True):
    import torch
    from codec.model import FACodec
    model = FACodec()
    model.eval()
    if pretrained:
        encoder_checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/ns3_facodec_encoder.bin")
        decoder_checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/ns3_facodec_decoder.bin")
        redecoder_checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/ns3_facodec_redecoder.bin")
        model.encoder.load_state_dict(encoder_checkpoint)
        model.decoder.load_state_dict(decoder_checkpoint)
        model.redecoder.load_state_dict(redecoder_checkpoint)
    return model
            