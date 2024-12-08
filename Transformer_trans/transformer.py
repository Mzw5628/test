import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import model
from model import Transformer

#数据处理部分
sentences = [
        ['我 想要 一瓶 啤酒 P', 'S i want a beer .', 'i want a beer . E'],
        ['我 想要 一瓶 可乐 P', 'S i want a coke .', 'i want a coke . E']
]

src_vocab = {'P' : 0, '我' : 1, '想要' : 2, '一瓶' : 3, '啤酒' : 4, '可乐' : 5}
model.src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
model.tgt_vocab_size = len(tgt_vocab)

src_len = 5
tgt_len = 6 


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
      enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
      dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
      dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] 

      enc_inputs.extend(enc_input)
      dec_inputs.extend(dec_input)
      dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs

  def __len__(self):
    return self.enc_inputs.shape[0]

  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

#模型部分参数
model.d_model = 512
model.d_ff = 2048
model.d_k = d_v = 64
model.n_layers = 6
model.n_heads = 8

model = Transformer().cuda()
if os.path.exists('transformer.pth'):
  model.load_state_dict(torch.load('transformer.pth',weights_only=True))
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

#训练部分
min_loss = float('inf')
for epoch in range(10):
    for enc_inputs, dec_inputs, dec_outputs in loader:

      enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()

      outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
      loss = criterion(outputs, dec_outputs.view(-1))
      print(f"Epoch: {epoch+1}, loss: {loss:.6f}")
      if loss < min_loss:
        min_loss = loss
        torch.save(model.state_dict(), 'transformer.pth')
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

#测试部分
def greedy_decoder(model, enc_input, start_symbol):
    
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
    return dec_input      

enc_inputs, _, _ = next(iter(loader))
enc_inputs = enc_inputs.cuda()

for i in range(len(enc_inputs)):
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print([idx2word[n.item()] for n in predict.squeeze()])