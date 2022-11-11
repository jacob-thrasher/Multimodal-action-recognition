from curses import resize_term
import torch
import torch.nn as nn
import math
import logging
from typing import Optional
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


# position wise encoding
class PositionalEncodingComponent(nn.Module):
    '''
    Class to encode positional information to tokens.
    For future, I want that this class to work even for sequences longer than 5000
    '''

    def __init__(self, hid_dim, dropout=0.2, max_len=5000):
        super().__init__()

        assert hid_dim % 2 == 0  # If not, it will result error in allocation to positional_encodings[:,1::2] later

        self.dropout = nn.Dropout(dropout)

        self.positional_encodings = nn.Parameter(torch.zeros(1, max_len, hid_dim), requires_grad=False)
        # Positional Embeddings : [1,max_len,hid_dim]

        pos = torch.arange(0, max_len).unsqueeze(1)  # pos : [max_len,1]
        div_term = torch.exp(-torch.arange(0, hid_dim, 2) * math.log(
            10000.0) / hid_dim)  # Calculating value of 1/(10000^(2i/hid_dim)) in log space and then exponentiating it
        # div_term: [hid_dim//2]

        self.positional_encodings[:, :, 0::2] = torch.sin(pos * div_term)  # pos*div_term [max_len,hid_dim//2]
        self.positional_encodings[:, :, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        # TODO: update this for very long sequences
        x = x + self.positional_encodings[:, :x.size(1)].detach()
        return self.dropout(x)


# feed forward
class FeedForwardComponent(nn.Module):
    '''
    Class for pointwise feed forward connections
    '''

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, x):
        # x : [batch_size,seq_len,hid_dim]
        x = self.dropout(torch.relu(self.fc1(x)))

        # x : [batch_size,seq_len,pf_dim]
        x = self.fc2(x)

        # x : [batch_size,seq_len,hid_dim]
        return x


# multi headed attention
class MultiHeadedAttentionComponent(nn.Module):
    '''
    Multiheaded attention Component.
    '''

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0  # Since we split hid_dims into n_heads

        self.hid_dim = hid_dim
        self.n_heads = n_heads  # no of heads in 'multiheaded' attention
        self.head_dim = hid_dim // n_heads  # dims of each head

        # Transformation from source vector to query vector
        self.fc_q = nn.Linear(hid_dim, hid_dim)

        # Transformation from source vector to key vector
        self.fc_k = nn.Linear(hid_dim, hid_dim)

        # Transformation from source vector to value vector
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        # Used in self attention for smoother gradients
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([self.head_dim])), requires_grad=False)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        # query : [batch_size, query_len, hid_dim]
        # key : [batch_size, key_len, hid_dim]
        # value : [batch_size, value_len, hid_dim]

        batch_size = query.shape[0]

        # Transforming quey,key,values
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q : [batch_size, query_len, hid_dim]
        # K : [batch_size, key_len, hid_dim]
        # V : [batch_size, value_len,hid_dim]

        # Changing shapes to acocmadate n_heads information
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q : [batch_size, n_heads, query_len, head_dim]
        # K : [batch_size, n_heads, key_len, head_dim]
        # V : [batch_size, n_heads, value_len, head_dim]

        # Calculating alpha
        score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # score : [batch_size, n_heads, query_len, key_len]

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e10)

        alpha = torch.softmax(score, dim=-1)
        # alpha : [batch_size, n_heads, query_len, key_len]

        # Get the final self-attention  vector
        x = torch.matmul(self.dropout(alpha), V)
        # x : [batch_size, n_heads, query_len, head_dim]

        # Reshaping self attention vector to concatenate
        x = x.permute(0, 2, 1, 3).contiguous()
        # x : [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x: [batch_size, query_len, hid_dim]

        # Transforming concatenated outputs
        x = self.fc_o(x)
        # x : [batch_size, query_len, hid_dim]

        return x, alpha


# EncodingLayer
class EncodingLayer(nn.Module):
    '''
    Operations of a single layer. Each layer contains:
    1) multihead attention, followed by
    2) LayerNorm of addition of multihead attention output and input to the layer, followed by
    3) FeedForward connections, followed by
    4) LayerNorm of addition of FeedForward outputs and output of previous layerNorm.
    '''

    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after self-attention
        self.ff_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after FeedForward component

        self.self_attention = MultiHeadedAttentionComponent(hid_dim, n_heads, dropout)
        self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src : [batch_size, src_len, hid_dim]
        # src_mask : [batch_size, 1, 1, src_len]

        # get self-attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # LayerNorm after dropout
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src : [batch_size, src_len, hid_dim]

        # FeedForward
        _src = self.feed_forward(src)

        # layerNorm after dropout
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src: [batch_size, src_len, hid_dim]

        return src

class AudioRepresentations(nn.Module):
    '''
    Group of layers that give final audio representation for cross attention

    The class get an input of size [batch_size,max_audio_len]
    we split the max_audio_len by audio_split_samples.
    Example: if the input was [10,60000] and audio_split_samples as 1000
    then we split the input as [10,60,1000]
    '''

    def __init__(self, audio_split_samples, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length):
        super().__init__()

        # Used for splitting the original signal
        self.audio_split_samples = audio_split_samples

        # Transform input from audio_split_dim to hid_dim
        self.transform_input = nn.Linear(audio_split_samples, hid_dim)

        self.pos_embedding = PositionalEncodingComponent(hid_dim, dropout, max_length)

        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

    def forward(self, audio):
        # You don't need mask for audio in attention because that padded
        # audio : [batch_size, max_audio_len]

        assert audio.shape[1] % self.audio_split_samples == 0

        batch_size = audio.shape[0]
        audio = audio.reshape(batch_size, -1, self.audio_split_samples)
        # audio : [batch_size, src_len , audio_split_samples]

        audio_embeddings = self.transform_input(audio) * self.scale
        # audio embeddings : [batch_size, src_len, hid_dim]

        # TODO: find better ways to give positional information. Here it is giving each audio_split_sample chunk same
        #  positional embedding
        audio = self.pos_embedding(audio_embeddings)
        # audio : [batch_size, src_len, hid_dim]

        for layer in self.layers:
            audio = layer(audio)
        # audio : [batch_size, src_len, hid_dim]

        return audio


class TextRepresentations(nn.Module):
    """
    Group of layers that give final text representation for cross attention
    """

    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, text_pad_index, max_length=5000):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = PositionalEncodingComponent(hid_dim, dropout, max_length)

        # encoder layers
        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

        self.text_pad_index = text_pad_index

    def create_text_mask(self, text):
        # masks padded values of text

        # text : [batch_size, src_len]
        text_mask = (text != self.text_pad_index).unsqueeze(1).unsqueeze(2)

        return text_mask

    def forward(self, text):
        # text : [batch_size, src_len]

        text_mask = self.create_text_mask(text)
        # text_mask : [batch_size,1,1,src_len]

        batch_size = text.shape[0]
        src_len = text.shape[1]

        tok_embeddings = self.tok_embedding(text) * self.scale

        # token plus position embeddings
        text = self.pos_embedding(tok_embeddings)

        for layer in self.layers:
            text = layer(text, text_mask)
        # src : [batch_size, src_len, hid_dim]

        return text

class SpecRepresentations(nn.Module):
    """
    Embeds spectrogram component via image patching and gives final representation for cross attention
    """
    def __init__(self, img_dim=(256, 512), p_dim=(16, 16), hid_dim=256, pf_dim=512, n_heads=4, n_layers=12, dropout=0):
        super().__init__()
        
        W, H = img_dim
        w, h = p_dim
        nw, nh = W // w, H // h
        self.hid_dim = hid_dim
        proj_dim = 3 * w * h

        self.embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=h, pw=w),
            nn.Linear(proj_dim, hid_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.pos_embedding = PositionalEncodingComponent(hid_dim=hid_dim, dropout=dropout, max_len=nh*nw)

        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, spec):
        spec = self.embedding(spec)
        spec = self.pos_embedding(spec)

        b, _, _ = spec.size()
        cls_to_batch = self.cls_token.expand([b, -1, -1])
        spec = torch.cat((cls_to_batch, spec), dim=1)
        
        for layer in self.layers:
            spec = layer(spec)

        return spec

class VideoRepresentations(nn.Module):
    """
    Embeds video component via tubelet embedding and gives final representation for cross attention
    """
    def __init__(self, input_dim=(256, 256, 32), p_dim=(64, 64, 8), hid_dim=256, n_layers=12, n_heads=4, pf_dim=512, dropout=0, device='cuda'):
        super().__init__()

        W, H, T = input_dim
        w, h, t = p_dim
        nw, nh, nt = W // w, H // h, T // t

        tubelet_dim = t * h * w * 3
        self.tubelet_embedding = nn.Sequential(
            Rearrange('b c (t pt) (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=t, ph=h, pw=w),
            nn.Linear(tubelet_dim, hid_dim)
        )
        # Note, this works for 32 frames, but not 64 for some reason. Need to mess with it
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, nh * nw, hid_dim)).repeat(1, nt, 1, 1).to(device)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))

        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)


    def forward(self, video):
        # Expects videos in shape: [b, c, t, h, w]
        video = self.tubelet_embedding(video)
        video += self.pos_embedding
        b, t, hw, d = video.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b t n d', b=b, t=t)
        video = torch.cat((cls_tokens, video), dim=2)
        
        for layer in self.layers:
            video = layer(video)

        return video

# Cross Attention Layer
class CrossAttentionLayer(nn.Module):
    '''
    This layer takes input the audio and text representations after they have been 
    passed through their respective Encoding layers. 
    The text representations will act as query
    the audio representations will be key and values.
    So this will take most important features from text representation based on the
    attention between audio and the text features.
    '''

    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after self-attention
        self.ff_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after FeedForward component

        self.self_attention = MultiHeadedAttentionComponent(hid_dim, n_heads, dropout)
        self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, video, audio):
        # text : [batch_size, text_len, hid_dim]
        # audio : [batch_size, audio_len, hid_dim

        # get self-attention
        _video, _ = self.self_attention(video, audio, audio)

        # LayerNorm after dropout
        video = self.self_attn_layer_norm(video + self.dropout(_video))
        # text : [batch_size, text_len, hid_dim]

        # FeedForward
        _video = self.feed_forward(video)

        # layerNorm after dropout
        video = self.ff_layer_norm(video + self.dropout(_video))
        # text: [batch_size, text_len, hid_dim]

        return video


# Model
class Model(nn.Module):
    """
    Model class
    We will use <sos> token for prediction of classes
    """

    def __init__(self, audio_split_samples, hid_dim, audio_representation_layers, n_heads, pf_dim, dropout, max_length, \
                 video_dim, p_dim, video_representation_layers, \
                 cross_attention_layers, \
                 output_dim, config):
        super().__init__()
        # self.audio_representations = AudioRepresentations(audio_split_samples, hid_dim, audio_representation_layers,
        #                                                   n_heads, pf_dim, dropout, max_length)
        self.audio_representations = SpecRepresentations(img_dim=(256, 512), p_dim=(16, 16), hid_dim=hid_dim, pf_dim=pf_dim,
                                                        n_heads=n_heads, n_layers=audio_representation_layers, dropout=dropout)
        self.video_representations = VideoRepresentations(input_dim=video_dim, p_dim=p_dim, hid_dim=hid_dim, n_layers=video_representation_layers,
                                                        n_heads=n_heads, pf_dim=pf_dim, dropout=dropout)

        self.cross_attention = nn.ModuleList(
            [CrossAttentionLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(cross_attention_layers)])

        self.feed_forward_1 = nn.Linear(hid_dim, output_dim)


        self.output_dim = output_dim


        self.loss_1 = nn.CrossEntropyLoss()
        self.loss_2 = nn.CrossEntropyLoss()
        self.loss_3 = nn.CrossEntropyLoss()

        self.config = config

    def forward(self, video, audio, label):
        # audio : [batch_size, max_audio_len]
        # text : [batch_size, src_len]

        audio = self.audio_representations(audio)
        # audio : [batch_size, audio_len, hid_dim] where audio_len= max_audio_len/audio_split_samples

        video = self.video_representations(video)
        # text : [batch_size, src_len, hid_dim]

        for layer in self.cross_attention:
            crs_attn_out = layer(video, audio)

        pred_token = crs_attn_out[:, 0, :]
        # pred_token : [batch_size, hid_dim]

        output = self.feed_forward_1(pred_token)
        loss = self.loss_1(output, label)

        # if 'mode' in self.config and self.config['mode'] == "weighted_loss":
        #     # weighted mean based on the total number of labels for actions object and position
        #     # since the number of labels for actions and positions are less they are reducing
        #     # the loss value to very low numbers.
        #     loss = (self.output_dim_1*loss_in_action + self.output_dim_2*loss_in_object + self.output_dim_3*loss_in_position)/\
        #            (self.output_dim_1 + self.output_dim_2 + self.output_dim_3)
        # else:
        #     loss = (loss_in_action + loss_in_object + loss_in_position) / 3

        pred = torch.argmax(output, -1)

        return {'loss': loss, 'pred': pred}
