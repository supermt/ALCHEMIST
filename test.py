import itertools
import pandas as pd
import re
from log_class import log_recorder

LOG_DIR = "./test_data2/"
report_csv = "report.csv"
LOG_file = "LOG"


# from log_class import log_recorder
COMPACTION_LOG_HEAD = "/compaction/compaction_job.cc:755"
FLUSH_LOG_BEGIN = "flush_started"
FLUSH_LOG_END = "flush_finished"
FLUSH_FILE_CREATEION = "table_file_creation"

def load_log_and_qps(log_file, ground_truth_csv):
    # load the data
    return log_recorder(log_file,ground_truth_csv)

data_set = load_log_and_qps(LOG_DIR+LOG_file, LOG_DIR+report_csv)


ms_to_sec  = 1000000
time_slice = 100000 # 100 miusec, 100,000ms
switch_ratio = ms_to_sec/time_slice

real_time_speed=data_set.qps_df
flush_jobs = data_set.flush_df
input_tuple=[0,0,12345,7] 
import numpy as np
bucket = []
distance = int(real_time_speed.tail(1)["microsecs_elapsed"] * switch_ratio)
for i in range(distance):
    bucket.append([])
# then we use a bucket sort idea to count down the rest things
for flush_job in data_set.flush_df.iloc():
    # indices = (int(flush_job["start_time"]/time_slice), flush_job["end_time"]/time_slice)
    start_index = int(flush_job["start_time"]/time_slice)
    end_index=int(flush_job["end_time"]/time_slice) + 1
    payload = round(flush_job["flush_size"] / (1024*1024) ,2) # change to MB will be easier to calculate
    job_id = flush_job["job"]

    if start_index >= len(bucket)-10 or end_index >= len(bucket)-5: # the tail part is not accurant
        break
    for bucket_element in bucket[start_index:end_index]:
        bucket_element.append([0,0,payload,job_id])

for compaction_job in data_set.compaction_df.iloc():
    start_index = int(compaction_job["start_time"]/time_slice)
    end_index=int(compaction_job["end_time"]/time_slice) + 1
    input_size = round(compaction_job["input_data_size"] / (1024*1024) ,2) # change to MB will be easier to calculate
    output_size = round(compaction_job["total_output_size"] / (1024*1024) ,2)
    job_id = compaction_job["job"]
    if start_index >= len(bucket)-10 or end_index >= len(bucket)-5: # the tail part is not accurant
        break
    compaction_tuple=[1,input_size,output_size,job_id]
    for bucket_element in bucket[start_index:end_index]:
        bucket_element.append(compaction_tuple)


plain_input_result = np.array(bucket)
output_array = real_time_speed["interval_qps"]
# create the transformer model

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        print(src)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
# tranfer the bucket array list into a tensor
temp_list = []
empty_tuple=[0,0,0,0]
max_length = max([len(x) for x in bucket])
for element in bucket:
    while len(element) < max_length:
        element.append(empty_tuple)

input_tensor = torch.tensor(bucket)
eval_tensor = torch.tensor(output_array)

# Although I totally don't understand what happened, but ... it successed
batch_size = 20 # every 20 time slice, in here is 2 seconds as a batch
eval_batch_size = 10 # evaluate on every one second
train_data = input_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_data = batchify(input_tensor,batch_size)
val_data = batchify(eval_tensor,eval_batch_size)
# print(train_data)

bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

# and ... eh ... here I understand these are the hyper arugments?
ntokens = max(max(data_set.compaction_df["job"]),max(data_set.flush_df["job"])) # the size of vocabulary, in here is the id of different jobs.
print(ntokens)
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
# learning methods
criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
import time

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(output_array)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(output_array)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)