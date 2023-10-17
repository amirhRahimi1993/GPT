import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 256
batch_size = 64
embed_size = 384
nnLinear_multihead = embed_size * 4
num_head = 6
learning_rate = 3e-4
n_layer= 6
train_size= 0.9
test_size = 1-train_size
max_iters = 5000
drop_out= 0.6
iter_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("input.txt","r",encoding='utf-8') as f:
    text = f.read()
TEXT = list(sorted(set(text)))
EncoDict = {ch:i for i,ch in enumerate(TEXT)}
DecoDict = {i:ch for i,ch in enumerate(TEXT)}
print(EncoDict)
encoding= lambda s: [EncoDict[c] for c in s]
decoding= lambda s:''.join(DecoDict[encode_index] for encode_index in s)

data = torch.tensor(encoding(text),dtype= torch.int8)
end_of_train = int(len(data) * train_size)
train_data = data[:end_of_train]
validation_data = data[end_of_train:]

def get_batch(split):
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data)- block_size,(batch_size,))
    text_input = torch.stack([data[i:i+block_size] for i in ix])
    text_target = torch.stack([data[i+1:i + block_size+1] for i in ix])
    text_input,text_target= text_input.to(device), text_target.to(device)
    return text_input,text_target

class Head(nn.Module):
    def __init__(self,embeding_size,head_size):
        self.key = nn.Linear(embeding_size,head_size)
        self.query = nn.Linear(embeding_size,head_size)
        self.value= nn.Linear(embeding_size,head_size)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.head_size = head_size
        self.dropout= nn.Dropout(drop_out)

    def forward(self,x):
        B,T,C= x.shape
        key = self.key(x)
        query= self.query(x)
        value= self.value(x)
        wei =  (self.query @ key.transpose(-2,-1))/ (self.head_size ** 0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei= self.dropout(wei)
        out= wei @ value
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self,n_layer,head_size,embed_size):
        self.multihead = nn.ModuleList([Head(embed_size,head_size) for _ in range(n_layer)])
        self.forward_attention = nn.Linear(embed_size,embed_size)
        self.dropout= nn.Dropout(drop_out)
    def forward(self,x):
        out = torch.cat([h(x) in self.multihead],dim=-1)
        out= self.dropout(self.forward_attention(out))
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed),
                                 nn.ReLU(),
                                 nn.Linear(n_embed* 4 , n_embed),
                                 nn.Dropout(drop_out))
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        assert embed_size % n_head == 0
        head_size = embed_size// n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = self.norm1(x)
        x= self.sa(x) + x
        x = self.norm2(x)
        x= self.ffwd(x) + x


# Let Glue all thing together :)
class GPTLanguageModel(nn.Module):
    def __init__(self,vocab_size,n_embed,n_head,num_layer):
        self.embedding= nn.Embedding(vocab_size,n_embed)
        self.positional_encoding = nn.Embedding(block_size,n_embed)
        self.blocks = nn.ModuleList(*[Block(n_embed,n_head) for _ in num_layer])
        self.ln_f= nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)

    def _init_weight(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    def forward(self,idx,target= None):
        B , T = idx.shape
        tok_embed = self.embedding(idx)
        pos_embed = self.positional_encoding(idx)
        x = tok_embed + pos_embed
        x= self.blocks(x)
        x= self.ln_f(x)
        logit= self.lm_head(x)
        if target is None:
            loss = None
        else:
            B,T,C= logit.shape
            logit= logit.view(B*T,C)
            target= target.view(B*T)
            loss= F.cross_entropy(logit,target)
        return logit,loss
    def generate(self,idx,max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:,-block_size:]
            logit,loss = self(idx_cond)
            logit= logit[:,-1,:]
            prob= F.softmax(logit,dim=-1)
            idx_next = torch.multinomial(prob,num_samples=1)
            idx= torch.cat((idx,idx_next),dim=-1)
        return idx
model = GPTLanguageModel()
m=model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, "M parameter")
@torch.no_grad()
def estimate_loss(eval_iters = 200):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split]= losses.mean()
    model.train()
    return out
optimizer= torch.optim.AdamW(model.parameters(),lr= learning_rate)

for iter in range(max_iters):
    if iter % iter_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decoding(m.generate(context,max_new_tokens=500)[0].tolist()))
