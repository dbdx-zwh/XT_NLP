import torch
import torch.nn as nn
import torch.nn.functional as F
import time as t

class Attention(nn.Module):
    def __init__(self, vocab_size, args):
        super(Attention, self).__init__()
        self.vocab_size = vocab_size
        self.args = args
        self.encode = nn.RNN(input_size=vocab_size, hidden_size=args.hidden_size, bidirectional=True, dropout=0.5)
        self.decode = nn.RNN(input_size=vocab_size, hidden_size=args.hidden_size, bidirectional=False, dropout=0.5)
        self.attn1 = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.attn2 = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.out = nn.Linear(2 * args.hidden_size, vocab_size)
    
    def forward(self, enc_inputs, enc_hidden, dec_inputs):
        # inputs:(batch_size, maxlen, vocab_size)
        enc_inputs = enc_inputs.transpose(0, 1)  # enc_inputs: [n_step(=n_step, time step), batch_size, n_class]
        dec_inputs = dec_inputs.transpose(0, 1)  # dec_inputs: [n_step(=n_step, time step), batch_size, n_class]
        enc_outputs, enc_hidden = self.encode(enc_inputs, enc_hidden)
        enc_outputs = self.attn2(enc_outputs)
        time_step = self.args.max_sentence_len
        predict = torch.empty([time_step, self.args.batch_size, self.vocab_size])

        for i in range(0, time_step):
            dec_hidden = torch.cat([enc_hidden[0], enc_hidden[1]], dim=1)
            dec_hidden = self.attn1(dec_hidden)
            dec_hidden = dec_hidden.unsqueeze(0)
            # print(dec_hidden.shape)
            dec_output, dec_hidden = self.decode(dec_inputs[i].unsqueeze(0), dec_hidden)
            # print(dec_output.shape, dec_hidden.shape, enc_outputs.shape)
            attn_weights = self.get_att_weight(dec_output, enc_outputs)
            dec_output = dec_output.transpose(0, 1).squeeze(1)
            context = attn_weights.bmm(enc_outputs.transpose(0, 1)).squeeze(1)
            # print(context.shape, dec_output.shape)
            # t.sleep(10)
            predict[i] = self.out(torch.cat((context, dec_output), 1))

        # print(predict.shape)
        return predict.transpose(0, 1)


    def get_att_weight(self, dec_output, enc_outputs):  # get attention weight one 'dec_output' with 'enc_outputs'
        batch_size = self.args.batch_size
        time_step = self.args.max_sentence_len
        attn_scores = torch.zeros(batch_size, time_step)  # attn_scores : [n_step]
        enc_outputs = enc_outputs.transpose(0, 1)
        dec_output = dec_output.transpose(0, 1)
        for i in range(0, batch_size):
            for j in range(0, time_step):
                attn_scores[i][j] = self.get_att_score(dec_output[i].unsqueeze(0), enc_outputs[i][j].unsqueeze(0))

        # Normalize scores to weights in range 0 to 1
        return F.softmax(attn_scores).view(batch_size, 1, -1)

    def get_att_score(self, dec_output, enc_output):  # enc_outputs [batch_size, num_directions(=1) * n_hidden]
        # score = self.attn2(enc_output)  # score : [batch_size, n_hidden]
        # print(dec_output.shape, score.shape)
        return torch.dot(dec_output.view(-1), enc_output.view(-1))  # inner product make scalar value