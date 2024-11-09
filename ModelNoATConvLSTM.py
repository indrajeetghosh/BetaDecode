import torch
import torch.nn as nn
import torch.nn.functional as F

#ConvLSTM implementation from ndrplz (https://raw.githubusercontent.com/ndrplz/ConvLSTM_pytorch/master/convlstm.py)
class ConvLSTMCell1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell1D, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, length):
        return (torch.zeros(batch_size, self.hidden_dim, length, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, length, device=self.conv.weight.device))


class ConvLSTMEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims):
        super(ConvLSTMEncoder, self).__init__()
        self.encoder_1 = ConvLSTMCell1D(input_dim=in_channels, hidden_dim=hidden_dims[0], kernel_size=3)
        self.encoder_2 = ConvLSTMCell1D(input_dim=hidden_dims[0], hidden_dim=hidden_dims[1], kernel_size=3)
        self.encoder_3 = ConvLSTMCell1D(input_dim=hidden_dims[1], hidden_dim=hidden_dims[2], kernel_size=3)

    def init_encoder_hidden(self, batch_size, length):
        return [
            self.encoder_1.init_hidden(batch_size, length),
            self.encoder_2.init_hidden(batch_size, length),
            self.encoder_3.init_hidden(batch_size, length)
        ]

    def encode(self, x, seq_len, hidden_states):
        h_t, c_t = hidden_states[0]
        h_t2, c_t2 = hidden_states[1]
        h_t3, c_t3 = hidden_states[2]

        for t in range(seq_len):
            h_t, c_t = self.encoder_1(x[:, t, :].unsqueeze(2), [h_t, c_t])
            h_t2, c_t2 = self.encoder_2(h_t, [h_t2, c_t2])
            h_t3, c_t3 = self.encoder_3(h_t2, [h_t3, c_t3])

        encoder_vector = h_t3.mean(dim=2)
        return encoder_vector


class ConvLSTMDecoder(nn.Module):
    def __init__(self, out_channels, hidden_dims):
        super(ConvLSTMDecoder, self).__init__()
        self.decoder_fc = nn.Linear(hidden_dims[2], hidden_dims[2])
        self.decoder_1 = ConvLSTMCell1D(input_dim=hidden_dims[2], hidden_dim=hidden_dims[1], kernel_size=3)
        self.decoder_2 = ConvLSTMCell1D(input_dim=hidden_dims[1], hidden_dim=hidden_dims[0], kernel_size=3)
        self.decoder_3 = ConvLSTMCell1D(input_dim=hidden_dims[0], hidden_dim=out_channels, kernel_size=3)
        self.output_conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def init_decoder_hidden(self, batch_size, length):
        return [
            self.decoder_1.init_hidden(batch_size, length),
            self.decoder_2.init_hidden(batch_size, length),
            self.decoder_3.init_hidden(batch_size, length)
        ]

    def decode(self, z, future_seq, hidden_states):
        z = self.decoder_fc(z).unsqueeze(-1).expand(-1, -1, future_seq)
        h_t, c_t = hidden_states[0]
        h_t2, c_t2 = hidden_states[1]
        h_t3, c_t3 = hidden_states[2]
        
        outputs = []

        for step in range(future_seq):
            h_t, c_t = self.decoder_1(z[:, :, step], [h_t, c_t])
            h_t2, c_t2 = self.decoder_2(h_t, [h_t2, c_t2])
            h_t3, c_t3 = self.decoder_3(h_t2, [h_t3, c_t3])
            outputs.append(h_t3)

        outputs = torch.stack(outputs, dim=2)
        return torch.sigmoid(self.output_conv(outputs))

    
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# Our no attemtion BetaVAE model --- make sure your in_channels -- match your no.of channels 
class BetaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=[32, 64, 128]):
        super(BetaVAE, self).__init__()

        self.encoder = ConvLSTMEncoder(in_channels, hidden_dims)
        self.decoder = ConvLSTMDecoder(in_channels, hidden_dims)
    
        self.fc_mu = nn.Linear(hidden_dims[2], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[2], latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x, future_seq=10):
        b, seq_len, num_channels = x.size()
        
        encoder_hidden_states = self.encoder.init_encoder_hidden(batch_size=b, length=num_channels)
        encoder_vector = self.encoder.encode(x, seq_len, encoder_hidden_states)
        mu = self.fc_mu(encoder_vector)
        logvar = self.fc_logvar(encoder_vector)
        z = reparameterize(mu, logvar)
        decoder_hidden_states = self.decoder.init_decoder_hidden(batch_size=b, length=num_channels)
        outputs = self.decoder.decode(z, future_seq, decoder_hidden_states)
        return outputs, mu, logvar
