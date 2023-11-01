import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence
class WordAttention(nn.Module):
    def __init__(self,vocab_size,emb_size,word_rnn_size,word_rnn_layers,word_attn_size,dropout):
        super(WordAttention,self).__init__()
        self.embedding = nn.Embedding(vocab_size,emb_size)
        self.word_rnn = nn.GRU(emb_size,word_rnn_size,num_layers=word_rnn_layers,bidirectional=True,dropout=dropout,batch_first=True)
        self.word_attention = nn.Linear(2*word_rnn_size,word_attn_size)
        self.word_context_vector = nn.Linear(word_attn_size,1,bias=False)
        self.dropout = nn.Dropout(dropout)
        