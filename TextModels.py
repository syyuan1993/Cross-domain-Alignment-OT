from sklearn.feature_extraction.text import CountVectorizer
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def BOW(corpus):
	'''
	:param corpus: input sentence
	:return: names of words, counted number
	bag of word model, ignoring EOS and all punctuation,
	ignoring all capital/lower case difference
	'''
	vectorizer= CountVectorizer()
	X = vectorizer.fit_transform(corpus)
	return vectorizer.get_feature_names(), X.toarray()


class BagOfWords():
	def __init__(self, size):
		self.vocab_size = size

	def calculate(self, sent, length):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		count = torch.zeros(len(length), self.vocab_size).to(device)
		for j in range(len(length)):
			for i in range(length[j]):
				count[j, sent[j,i].long()]+=1
		count = count/length
		return count


class WordEmbedding(nn.Module):
	def __init__(self, embedding):
		super(WordEmbedding, self).__init__()
		# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.embedding = nn.Embedding.from_pretrained(embedding)

	def forward(self, sent):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		return self.embedding(torch.LongTensor(sent.to('cpu'))).to(device)


class LabelEmbedding(nn.Module):
	def __init__(self, embedding):
		super(WordEmbedding, self).__init__()
		# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.embedding = nn.Embedding.from_pretrained(embedding)

	def forward(self, sent):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		return self.embedding(torch.LongTensor(sent.to('cpu'))).to(device)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)#.flatten_parameters()

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(x)
        # cap_emb = out
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb.transpose(2,1)


if __name__=='__main__':
    model = BagOfWords(24272)
    count = model.calculate(torch.Tensor([1,34,2,4,4,5]))
    print(count[:10])



