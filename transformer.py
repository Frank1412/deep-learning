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
source_word_index = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
target_word_index = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
# source_word_index = {word: index for index, word in enumerate(set((sentences[0][0] + sentences[1][0]).split(" ")))}
# target_word_index = {word: index for index, word in enumerate(set((" ".join(sentences[0][1:] + sentences[1][1:])).split(" ")))}
source_vocab_size = len(source_word_index)
target_vocab_size = len(target_word_index)
print(source_word_index)
print(target_word_index)

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
    for i in range(len(encode_token)):
        if len(encode_token[i]) <= source_vocab_size:
            encode_token[i] += [0] * (source_vocab_size - len(encode_token[i]))
        else:
            encode_token[i] = encode_token[i][:source_vocab_size]
        if len(decode_token[i]) <= target_vocab_size:
            decode_token[i] += [0] * (target_vocab_size - len(decode_token[i]))
        else:
            decode_token[i] = decode_token[i][:target_vocab_size]
        if len(target_token[i]) <= target_vocab_size:
            target_token[i] += [0] * (target_vocab_size - len(target_token[i]))
        else:
            target_token[i] = target_token[i][:target_vocab_size]

    return torch.LongTensor(encode_token), torch.LongTensor(decode_token), torch.LongTensor(target_token)


encode_input, decode_input, target_input = make_data(sentences)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, embedding_dim)
        self.pe[:, 0::2] = torch.tensor([[torch.sin(pos / torch.pow(10000, i / embedding_dim)) for i in torch.arange(0, embedding_dim, 2)] for pos in
                                         torch.arange(max_len, dtype=torch.float)])
        self.pe[:, 1::2] = torch.tensor([[torch.cos(pos / torch.pow(10000, i / embedding_dim)) for i in torch.arange(1, embedding_dim, 2)] for pos in
                                         torch.arange(max_len, dtype=torch.float)])

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
    self_attention mask 填充0 消除对SoftMax的影响
    inputQ, inputK: [batch_size, seq_len]
    """
    batch_size = inputQ.shape[0]
    seq_lenQ = inputQ.shape[1]
    seq_lenK = inputK.shape[1]

    mask = inputK.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_lenK]      tensor.data.eq(0) tensor中等于0的为True, 其他为False
    pad_mask = mask.expand(batch_size, seq_lenQ, seq_lenK)
    return pad_mask


def target_sequence_mask(seq):
    """
    seq: [batch_size, target_len, target_len]
    decoder sequence mask 训练时消除后入的词对前面的影响
    """
    attn_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]
    seq_mask = torch.triu(torch.ones(attn_shape), diagonal=1)   # torch.triu diagonal=1 下三角包括对角线全0矩阵
    return seq_mask.byte()


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

        Z = torch.matmul(F.softmax(QKt.masked_fill_(self_attention_mask.unsqueeze(1).repeat(1, n_head, 1, 1), 1e-10), dim=-1),
                         V)  # [batch_size, n_head, seq_len, qkv_dim]

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
    n_head=8, n_layer=6,
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

        # print(embedding.shape, position_encoding.shape)
        enc_outputs = embedding + position_encoding  # [batch_size, seq_len, embedding_dim]
        # print(attn_mask.shape)

        for i in range(n_layer):
            attn_out = self.self_attention(enc_outputs, enc_outputs, enc_outputs, attn_mask)

            # Feed Forward
            ff_out = self.ff(attn_out)  # [batch_size, seq_len, embedding_dim]

            enc_outputs = ff_out  # [batch_size, seq_len, embedding_dim]
            # print(attn_out.shape, enc_outputs.shape)

        return enc_outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.self_attention = MultiHeadAttention()
        self.ff = FeedForward()
        self.fc = nn.Linear(embedding_dim, target_vocab_size, bias=False)

    def forward(self, enc_inputs, enc_outputs, dec_inputs):
        embedding = self.embedding(dec_inputs)
        pe = PositionalEncoding(dec_inputs.shape[1])
        position_encoding = pe(dec_inputs.size(0), dec_inputs.shape[1])  # [batch_size, seq_len, embedding_dim]
        dec_outputs = embedding + position_encoding

        # self_enc_attn_mask
        self_enc_attn_mask = padding_mask(dec_inputs, enc_inputs)  # [batch_size, target_len, source_len]

        # self_dec_attn mask
        self_attn_mask = padding_mask(dec_inputs, dec_inputs)

        # sequence_mask
        seq_mask = target_sequence_mask(dec_inputs)    # [batch_size, target_len, target_len]

        dec_attn_mask = torch.gt((seq_mask + self_attn_mask), 0)    # 每个元素和0比较，大于0返回True

        for i in range(n_layer):
            print(dec_inputs.shape)

            dec_outputs = self.self_attention(dec_outputs, dec_outputs, dec_outputs, dec_attn_mask)

            dec_outputs = self.self_attention(dec_outputs, enc_outputs, enc_outputs, self_enc_attn_mask)

            dec_outputs = self.ff(dec_outputs)  # [batch_size, target_len, embedding_dim]

        out = torch.softmax(self.fc(dec_outputs), dim=-1)

        return out


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        out = self.decoder(enc_inputs, enc_outputs, dec_inputs)
        return out


model = Transformer()
result = model(encode_input, decode_input)
print(result)
# enc = Encoder()
# enc(encode_input)
