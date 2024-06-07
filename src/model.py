import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

class myLSTM(nn.Module):
    """
    My lstm for text classification
    """
    def __init__(self, config):
        super(myLSTM, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

        self.hidden_size = self.bert.config.hidden_size
        config.num_labels = 3
        self.output_size = config.num_labels
        self.dropout = nn.Dropout(0.2)

        hidden_size = self.hidden_size
        output_size = self.output_size
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.lin = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.atten_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size, 1),
        )
        hidden_size_low = 100
        
        self.mention_linear = nn.Sequential(
            nn.Linear( hidden_size_low+ config.pos_emb_size * 2 + config.sememe_emb_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )

        self.mention_linear_no_pos = nn.Sequential(
            nn.Linear(hidden_size_low, hidden_size_low),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_low, output_size)
        )
        
        self.linear = nn.Linear(hidden_size_low * 4, output_size - 1)
        self.linear_1 = nn.Linear(hidden_size, hidden_size_low)
        self.fuse_1 = nn.Linear(hidden_size_low, hidden_size_low, bias=False)
        self.fuse_2 = nn.Linear(hidden_size_low, hidden_size_low)

    def get_mention_indices(self, outputs):
        if len(outputs.shape) > 1:
            outputs = outputs.argmax(-1)
        bio_dict = self.config.id2tag
        outputs = [outputs]
        outputs = [[bio_dict[w.item()] for w in out] for out in outputs]

        indices = []
        length = 0
        for output in outputs:
            start, end = -1, -1
            for i, out in enumerate(output):
                index = i + length
                if out == 'B':
                    if start != -1:
                        indices.append((start, end))
                    start, end = index, index
                elif out == 'O':
                    if start != -1:
                        indices.append((start, end))
                    start, end = -1, -1
                else:
                    end = index
            if start != -1:
                indices.append((start, end))
            length += len(output)
        return indices
    
    def get_mention_emb(self, lstm_out, mention_index):
        mention_emb_list = []
        mention_start, mention_end = zip(*mention_index)
        mention_start = torch.tensor(mention_start).to(self.config.device)
        mention_end = torch.tensor(mention_end).to(self.config.device)
        mention_emb_list.append(lstm_out.index_select(0, mention_start))
        mention_emb_list.append(lstm_out.index_select(0, mention_end))
        
        mention_emb = torch.cat(mention_emb_list, 1)
        return mention_emb

    def get_mention_labels(self, predict_indices, gold_sets):

        mention_matrix = torch.zeros(len(predict_indices), len(predict_indices)).long().to(self.config.device)
        indices_dict = {w : i for i, w in enumerate(predict_indices)}
        for i in range(len(predict_indices)):
            mention_matrix[i, i] = 1
            pass
        for all_set in gold_sets:
            for gold_set in all_set:
                for mention_0 in gold_set:
                    if mention_0 not in indices_dict:
                        continue
                    for mention_1 in gold_set:
                        if mention_1 not in indices_dict:
                            continue
                        s1, s2 = indices_dict[mention_0], indices_dict[mention_1]
                        mention_matrix[s1, s2] = 1
                        mention_matrix[s2, s1] = 1
        return mention_matrix
    
    def forward(self, **kwargs):
        input_ids, input_masks, input_segments = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments']]
        input_labels = kwargs['bio_list']
        mention_sets = kwargs['chains']
        seq_indicator = kwargs['seq_indicator']

        output = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0]
        output = output[seq_indicator.bool()]
        output = self.dropout(output)
        criterion = nn.CrossEntropyLoss()
        output = self.linear_1(output)

        labels = input_labels[0]

        flatten_output = output

        flatten_output_nomask = self.dropout(flatten_output)

        predict_output = self.mention_linear_no_pos(flatten_output)

        losses = criterion(predict_output, labels)

        predict_indices = self.get_mention_indices(predict_output)
        gold_indices = self.get_mention_indices(labels)
        if len(predict_indices) == 0:
            predict_indices = gold_indices[:1]

        mention_emb = self.get_mention_emb(flatten_output_nomask, predict_indices)
        mention_label = self.get_mention_labels(predict_indices, mention_sets)

        mention_emb_r = mention_emb.unsqueeze(1)
        mention_emb_c = mention_emb.unsqueeze(0)

        mention_emb_agg = torch.cat((mention_emb_c * mention_emb_r, mention_emb_r + mention_emb_c), -1)

        mention_interaction = self.linear(mention_emb_agg)
        new_mention_interaction, new_mention_label = [], []
        for i in range(mention_interaction.shape[0]):
            new_mention_interaction.append(mention_interaction[i, :i + 1])
            new_mention_label.append(mention_label[i, :i + 1])
        new_mention_label = torch.cat(new_mention_label)
        new_mention_interaction = torch.cat(new_mention_interaction, 0)
        criterion = nn.CrossEntropyLoss()
        tmp_loss = criterion(new_mention_interaction, new_mention_label)
        losses += tmp_loss

        return losses, predict_output, labels, predict_indices, mention_interaction
    
    def inference(self, **kwargs):
        input_ids, input_masks, input_segments = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments']]
        seq_indicator = kwargs['seq_indicator']

        output = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0]
        output = output[seq_indicator.bool()]
        output = self.dropout(output)
        output = self.linear_1(output)

        flatten_output = output

        flatten_output_nomask = self.dropout(flatten_output)

        predict_output = self.mention_linear_no_pos(flatten_output)

        predict_indices = self.get_mention_indices(predict_output)
        if len(predict_indices) == 0:
            predict_indices = [(0, 1)]

        mention_emb = self.get_mention_emb(flatten_output_nomask, predict_indices)

        mention_emb_r = mention_emb.unsqueeze(1)
        mention_emb_c = mention_emb.unsqueeze(0)

        mention_emb_agg = torch.cat((mention_emb_c * mention_emb_r, mention_emb_r + mention_emb_c), -1)

        mention_interaction = self.linear(mention_emb_agg)
        new_mention_interaction = []
        for i in range(mention_interaction.shape[0]):
            new_mention_interaction.append(mention_interaction[i, :i + 1])
        new_mention_interaction = torch.cat(new_mention_interaction, 0)

        return predict_output, predict_indices, mention_interaction