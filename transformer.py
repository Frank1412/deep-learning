import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]
source_word_index = {word: index for index, word in enumerate(set((sentences[0][0] + sentences[1][0]).split(" ")))}
target_word_index = {word: index for index, word in enumerate(set((" ".join(sentences[0][1:] + sentences[1][1:])).split(" ")))}
source_vocab_size = len(source_word_index)
target_vcab_size = len(target_word_index)

seq_len = 10
embedding_dim = 128
qkv_dim = 64
n_head = 8  # number of self-MultiHeadAttention
n_layer = 6  # number of Encoder and Decoder Layer
d_ff = 2048  # FeedForward dimension


def make_data(sentence):
    encode_token = []
    decode_token = []
    target_token = []
    for s in sentence:
        encode_token.append([source_word_index[word] for word in s[0].split(" ")])
        decode_token.append([target_word_index[word] for word in s[1].split(" ")])
        target_token.append([target_word_index[word] for word in s[2].split(" ")])

    return torch.LongTensor(encode_token), torch.LongTensor(decode_token), torch.LongTensor(target_token)


encode_input, decode_input, target_input = make_data(sentences)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, embedding_dim)
        self.pe[:, 0::2] = torch.tensor([[torch.sin(pos / torch.pow(10000, i / embedding_dim)) for i in torch.arange(0, embedding_dim, 2)] for pos in torch.arange(max_len, dtype=torch.float)])
        self.pe[:, 1::2] = torch.tensor([[torch.cos(pos / torch.pow(10000, i / embedding_dim)) for i in torch.arange(1, embedding_dim, 2)] for pos in torch.arange(max_len, dtype=torch.float)])

    def forward(self, batch_size, max_len):
        return self.pe.expand(batch_size, self.pe.size(0), self.pe.size(1))


class MyDataSet(data.Dataset):
    def __init__(self, enc_input, out_input, tgt_input):
        self.enc_input = enc_input
        self.out_input = out_input
        self.tgt_input = tgt_input

    def __len__(self):
        return self.enc_input.shape[0]

    def __getitem__(self, idx):
        return self.enc_input[idx], self.out_input[idx], self.tgt_input[idx]


def padding_mask(inputQ, inputK):
    """
    inputQ, inputK: [batch_size, seq_len]
    """
    batch_size = inputQ.shape[0]
    seq_lenQ = inputQ.shape[1]
    seq_lenK = inputK.shape[1]

    mask = inputQ.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]      tensor.data.eq(0) tensor中等于0的为True, 其他为False
    pad_mask = mask.expand(batch_size, seq_lenK, seq_lenQ)
    return pad_mask


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(embedding_dim, qkv_dim * n_head, bias=False)
        self.WK = nn.Linear(embedding_dim, qkv_dim * n_head, bias=False)
        self.WV = nn.Linear(embedding_dim, qkv_dim * n_head, bias=False)

        self.fc = nn.Linear(qkv_dim * n_head, embedding_dim, bias=False)

    def forward(self, inputQ, inputK, inputV, self_attention_mask):
        """
        inputQ, inputK, inputV: [batch_size, seq_len, embedding_dim]
        Q,K,V: [batch_size, seq_len, qkv_dim]
        attention_mask: [batch_size, seq_lenQ, seq_lenK]
        """
        batch_size = inputQ.shape[0]
        Q = self.WQ(inputQ).unsqueeze(1).reshape(batch_size, n_head, -1, qkv_dim)  # [batch_size, n_head, seq_len, qkv_dim]
        K = self.WQ(inputK).unsqueeze(1).reshape(batch_size, n_head, -1, qkv_dim)
        V = self.WQ(inputV).unsqueeze(1).reshape(batch_size, n_head, -1, qkv_dim)

        QKt = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, n_head, seq_len, qkv_dim]

        Z = torch.matmul(F.softmax(QKt.masked_fill_(self_attention_mask.unsqueeze(1).repeat(1, n_head, 1, 1), 1e-10), dim=-1), V)  # [batch_size, n_head, seq_len, qkv_dim]

        self_attn_out = self.fc(Z.reshape(batch_size, -1, qkv_dim * n_head))  # [batch_size, seq_len, embedding_dim]

        # add + normal
        out = nn.LayerNorm(self_attn_out.shape[-1:])(inputQ + self_attn_out)

        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, embedding_dim, bias=False)

    def forward(self, attn_out):
        """

        :param attn_out: [batch_size, seq_len, embedding_dim]
        :return:
        """
        out = self.fc1(attn_out)
        out = self.fc2(out)
        out = nn.LayerNorm(attn_out.shape[-1])(attn_out + out)
        return out


class Encoder(nn.Module):
    """
    n_head=8, n_layer=6
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(source_vocab_size, embedding_dim)
        # self.positional_encoding = PositionalEncoding()
        self.self_attention = MultiHeadAttention()
        self.ff = FeedForward()

    def forward(self, enc_inputs):
        embedding = self.embedding(enc_inputs)  # [batch_size, seq_len, embedding_dim]
        # print(enc_inputs.shape)

        pe = PositionalEncoding(enc_inputs.shape[1])
        position_encoding = pe(enc_inputs.size(0), enc_inputs.shape[1])  # [batch_size, seq_len, embedding_dim]
        # print(position_encoding[0,3:6,:])

        # padding_mask
        attn_mask = padding_mask(enc_inputs, enc_inputs)

        print(embedding.shape, position_encoding.shape)
        enc_outputs = embedding + position_encoding  # [batch_size, seq_len, embedding_dim]
        print(attn_mask.shape)

        for i in range(n_layer):

            attn_out = self.self_attention(enc_outputs, enc_outputs, enc_outputs, attn_mask)

            # Feed Forward
            ff_out = self.ff(attn_out)  # [batch_size, seq_len, embedding_dim]

            enc_outputs = ff_out  # [batch_size, seq_len, embedding_dim]
            print(attn_out.shape, enc_outputs.shape)

        return


enc = Encoder()
enc(encode_input)
