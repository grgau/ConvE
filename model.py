import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = torch.sigmoid(pred)

        return pred



class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        # self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        # self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        # self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        # self.loss = torch.nn.BCELoss()
        # self.emb_dim1 = args.embedding_shape1
        # self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        # self.fc = torch.nn.Linear(args.hidden_size,args.embedding_dim)
        # print(num_entities, num_relations)

    # def init(self):
    #     xavier_normal_(self.emb_e.weight.data)
    #     xavier_normal_(self.emb_rel.weight.data)

    def forward(self, stacked_inputs):
        # e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        # rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        # stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        # x = self.fc(x)
        # x = self.hidden_drop(x)
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        # x += self.b.expand_as(x)
        # pred = torch.sigmoid(x)

        # return pred
        return x

class Combine(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Combine, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.loss = torch.nn.BCELoss()
        self.batch_size = args.batch_size
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.cnn = ConvE(args, num_entities, num_relations)
        self.rnn = torch.nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(args.hidden_size,args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        x = self.cnn(stacked_inputs)

        print("pos cnn")
        print(x.shape)

        x = x.view(self.batch_size, 1, -1)

        print("pos first view")
        print(x.shape)

        x, (h_n, h_c) = self.rnn(x)

        print("pos rnn")
        print(x.shape)

        x = x.view(x.shape[0], -1)

        print("pos second view")
        print(x.shape)

        x = self.fc(x)

        print("pos fully conected")
        print(x.shape)

        x = F.log_softmax(x, dim=1)

        print("pos softmax")
        print(x.shape)

        x = torch.mm(x, self.emb_e.weight.transpose(1,0))

        print("pos mm")
        print(x.shape)

        x += self.b.expand_as(x)

        print("pos expand")
        print(x.shape)

        pred = torch.sigmoid(x)

        print("pos pred")
        print(pred)

        return pred
        # return x

        # e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        # rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        # print(e1_embedded)
        # print(rel_embedded)                

        # batch_size, timesteps, C, H, W = x.size()
        # c_in = x.view(batch_size * timesteps, C, H, W)
        # c_out = self.cnn(c_in)
        # r_in = c_out.view(batch_size, timesteps, -1)
        # r_out, (h_n, h_c) = self.rnn(r_in)
        # r_out2 = self.linear(r_out[:, -1, :])

class Lstm(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Lstm, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.loss = torch.nn.BCELoss()
        self.batch_size = args.batch_size
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.rnn = torch.nn.LSTM(input_size=400, hidden_size=args.hidden_size, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(args.hidden_size,args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        
    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        # print("stacked")
        # print(stacked_inputs.shape)

        x = stacked_inputs.view(128, 1, -1)
        x, (hn, cn) = self.rnn(x)
        x = self.fc(x[:, -1, :])

        # print("valor de x pós fc")
        # print(x)
        # print(x.shape)
        # print(x.type)

        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        # print("pred")
        # print(pred.shape)
        # print(pred.type)

        return pred
