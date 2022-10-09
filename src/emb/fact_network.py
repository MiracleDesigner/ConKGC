import torch
import torch.nn as nn
import torch.nn.functional as F

class ConKGC(nn.Module):
    def __init__(self, args):
        super(ConKGC, self).__init__()
        self.args = args
        self.num_emb_hidden_layers = args.num_emb_hidden_layers
        self.emb_hidden_size = args.emb_hidden_size
        self.num_hidden_layers = args.num_hidden_layers
        self.hidden_size = args.hidden_size
        self.hidden_act = 'relu'
        self.activation = getattr(F, self.hidden_act)
        self.embedding_dim = args.entity_dim
        self.hidden_dropout = nn.Dropout(args.hidden_dropout_rate)
        if args.use_tau:
            # self.inv_lambda = nn.Parameter(torch.tensor(1.0 / args.lamb), requires_grad=True)
            # self.inv_tau = nn.Parameter(torch.tensor(1.0 / args.tau), requires_grad=True)
            # self.inv_lambda = 1.0 / args.lamb
            self.inv_tau = 1.0 / args.tau

        self.layers = nn.ModuleList()
        self.entity_layers = nn.ModuleList()
        self.relation_layers = nn.ModuleList()

        # embedding network
        if self.num_emb_hidden_layers == 1:
            self.entity_layers.append(nn.Linear(self.embedding_dim, self.emb_hidden_size))
            self.relation_layers.append(nn.Linear(self.embedding_dim, self.emb_hidden_size))
        else:
            for i in range(self.num_emb_hidden_layers):
                if i == 0:
                    self.entity_layers.append(nn.Linear(self.embedding_dim, self.emb_hidden_size))
                    self.relation_layers.append(nn.Linear(self.embedding_dim, self.emb_hidden_size))
                else:
                    self.entity_layers.append(nn.Linear(self.emb_hidden_size, self.emb_hidden_size))
                    self.relation_layers.append(nn.Linear(self.emb_hidden_size, self.emb_hidden_size))

        # predict network
        if self.num_hidden_layers == 1:
            self.layers.append(nn.Linear(2 * self.emb_hidden_size, self.hidden_size))
        else:
            for i in range(self.num_hidden_layers):
                if i == 0:
                    self.layers.append(nn.Linear(2 * self.emb_hidden_size, self.hidden_size))
                else:
                    self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        # Hidden-to-output
        self.layers.append(nn.Linear(self.hidden_size, self.embedding_dim))

    def forward(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_all_entity_embeddings()

        for i, layer in enumerate(self.entity_layers):
            # at least one
            E1 = layer(E1)
            if i != len(self.entity_layers) - 1:
                # No activation/dropout for final layer
                E1 = self.activation(E1)
                E1 = self.hidden_dropout(E1)

        for i, layer in enumerate(self.relation_layers):
            # at least one
            R = layer(R)
            if i != len(self.relation_layers) - 1:
                # No activation/dropout for final layer
                R = self.activation(R)
                R = self.hidden_dropout(R)

        S = torch.cat((E1 ,R), dim=1)
        for i, layer in enumerate(self.layers):
            # at least one
            S = layer(S)
            if i != len(self.layers) - 1:
                # No activation/dropout for final layer
                S = self.activation(S)
                S = self.hidden_dropout(S)

        S = torch.mm(S, E2.transpose(1, 0))
        if self.args.use_tau:
            S *= self.inv_tau
        S = torch.softmax(S, dim=1)
        return S

    def forward_train(self, e1, r, e2, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_all_entity_embeddings()

        for i, layer in enumerate(self.entity_layers):
            # at least one
            E1 = layer(E1)
            if i != len(self.entity_layers) - 1:
                # No activation/dropout for final layer
                E1 = self.activation(E1)
                E1 = self.hidden_dropout(E1)

        for i, layer in enumerate(self.relation_layers):
            # at least one
            R = layer(R)
            if i != len(self.relation_layers) - 1:
                # No activation/dropout for final layer
                R = self.activation(R)
                R = self.hidden_dropout(R)

        S = torch.cat((E1 ,R), dim=1)
        for i, layer in enumerate(self.layers):
            # at least one
            S = layer(S)
            if i != len(self.layers) - 1:
                # No activation/dropout for final layer
                S = self.activation(S)
                S = self.hidden_dropout(S)

        S = torch.mm(S, E2.transpose(1, 0))
        S = S*e2*self.inv_lambda + S*(1-e2)*self.inv_tau
        S = torch.softmax(S, dim=1)
        return S


