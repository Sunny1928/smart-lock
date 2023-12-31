import torch
from torch import nn
import utils.myconfig as myconfig

class BaseSpeakerEncoder(nn.Module):
    def _load_from(self, saved_model):
        var_dict=torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(var_dict["encoder_state_dict"])


class LstmSpeakerEncoder(BaseSpeakerEncoder):

    def __init__(self, saved_model=""):
        super(LstmSpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=myconfig.N_MFCC,
            hidden_size=myconfig.LSTM_HIDDEN_SIZE,
            num_layers=myconfig.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=myconfig.BI_LSTM)
        if saved_model:
            self._load_from(saved_model)


    def _aggregate_frames(self, batch_output):
        if myconfig.FRAME_AGGREGATION_MEAN:
            return torch.mean(batch_output, dim=1, keepdim=False)
        else:
            return batch_output[:, -1, :]


    def forward(self, x):
        D = 2 if myconfig.BI_LSTM else 1
        h0 = torch.zeros(D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE).to(myconfig.DEVICE)
        c0 = torch.zeros(D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE).to(myconfig.DEVICE)
        y, (hn, cn) = self.lstm(x, (h0, c0))
        return self._aggregate_frames(y)


def get_speaker_encoder(load_from=""):
    """Create speaker encoder model or load it from a saved model."""
    
    return LstmSpeakerEncoder(load_from).to(myconfig.DEVICE)







