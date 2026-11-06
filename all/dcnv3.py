import torch
from torch import nn


class ExponentialCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=True,
                 net_dropout=0.1):
        super(ExponentialCrossNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        self.gates = nn.ModuleList()
        
        self.input_dim = input_dim
        self.half = input_dim // 2

        for i in range(num_cross_layers):
            w_layer = nn.Linear(input_dim, self.half, bias=False).to(self.device)
            self.w.append(w_layer)
            self.b.append(nn.Parameter(torch.zeros((input_dim,), device=self.device)))

            gate_layer = nn.Linear(self.half, self.half).to(self.device)
            self.gates.append(gate_layer)

            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(self.half).to(self.device))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(self.half).to(self.device))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout).to(self.device))
            nn.init.uniform_(self.b[i].data)
            
        self.masker = nn.ReLU().to(self.device)

        self.dfc = nn.Linear(input_dim, 1).to(self.device)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.to(self.device)

        for i in range(self.num_cross_layers):
            H = self.w[i](x)

            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)

            gate = torch.sigmoid(self.gates[i](H))
            H = H * gate 

            H = torch.cat([H, H * mask], dim=-1)

            if H.shape[-1] != self.input_dim:
                pad_size = self.input_dim - H.shape[-1]
                pad = H.new_zeros(H.size(0), pad_size)
                H = torch.cat([H, pad], dim=-1)
            x = x * (H + self.b[i]) + x
            x = torch.clamp(x, min=-10, max=10)
            x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-12)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        logit = self.dfc(x)
        return logit


class LinearCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=True,
                 net_dropout=0.1):
        super(LinearCrossNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        self.gates = nn.ModuleList()

        self.input_dim = input_dim
        self.half = input_dim // 2
        
        for i in range(num_cross_layers):
            w_layer = nn.Linear(input_dim, self.half, bias=False).to(self.device)
            self.w.append(w_layer)
            self.b.append(nn.Parameter(torch.zeros((input_dim,), device=self.device)))

            gate_layer = nn.Linear(self.half, self.half).to(self.device)
            self.gates.append(gate_layer)
            
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(self.half).to(self.device))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(self.half).to(self.device))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout).to(self.device))
            nn.init.uniform_(self.b[i].data)
            
        self.masker = nn.ReLU().to(self.device)

        self.sfc = nn.Linear(input_dim, 1).to(self.device)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.to(self.device)
        x0 = x

        for i in range(self.num_cross_layers):
            H = self.w[i](x)
    
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)

            gate = torch.sigmoid(self.gates[i](H))
            H = H * gate        

            H = torch.cat([H, H * mask], dim=-1)
            if H.shape[-1] != self.input_dim:
                pad_size = self.input_dim - H.shape[-1]
                pad = H.new_zeros(H.size(0), pad_size)
                H = torch.cat([H, pad], dim=-1)

            x = x0 * (H + self.b[i]) + x
            x = torch.clamp(x, min=-10, max=10)
            x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-12)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
                
        logit = self.sfc(x)
        return logit

class DCNv3(nn.Module):
    def __init__(self,
                 input_dim,
                 num_deep_cross_layers=2,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.1,
                 layer_norm=True,
                 batch_norm=True):
        super(DCNv3, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.ECN = ExponentialCrossNetwork(
            input_dim=input_dim,
            num_cross_layers=num_deep_cross_layers,
            net_dropout=deep_net_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm
        ).to(self.device)
        
        self.LCN = LinearCrossNetwork(
            input_dim=input_dim,
            num_cross_layers=num_shallow_cross_layers,
            net_dropout=shallow_net_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm
        ).to(self.device)
        
        self.apply(self._init_weights)
        self.output_activation = torch.sigmoid

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        
        feature_emb = inputs
        dlogit = self.ECN(feature_emb).mean(dim=1)
        slogit = self.LCN(feature_emb).mean(dim=1)
        logit = (dlogit + slogit) * 0.5
    
        y_pred = self.output_activation(logit)
        y_d = self.output_activation(dlogit)
        y_s = self.output_activation(slogit)
        return {
            "y_pred": y_pred,
            "y_d": y_d,
            "y_s": y_s
        }

class TriBCE_Loss(nn.Module):
    def __init__(self):
        super(TriBCE_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true, y_d, y_s):
        loss = self.bce_loss(y_pred, y_true)
        loss_d = self.bce_loss(y_d, y_true)
        loss_s = self.bce_loss(y_s, y_true)
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros(1).to(weight_d.device))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros(1).to(weight_s.device))
        loss = loss + loss_d * weight_d + loss_s * weight_s
        return loss

class Weighted_TriBCE_Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(Weighted_TriBCE_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true, y_d, y_s):
        pos_count = torch.sum(y_true)
        neg_count = y_true.numel() - pos_count

        pos_count = torch.clamp(pos_count, min=1.0)
        neg_count = torch.clamp(neg_count, min=1.0)

        pos_weight = neg_count / (pos_count + neg_count)
        neg_weight = pos_count / (pos_count + neg_count)

        sample_weight = torch.where(y_true == 1, pos_weight, neg_weight)

        bce_loss = nn.BCELoss(weight=sample_weight, reduction=self.reduction)

        loss = bce_loss(y_pred, y_true)
        loss_d = bce_loss(y_d, y_true)
        loss_s = bce_loss(y_s, y_true)

        weight_d = torch.where(loss_d > loss, loss_d - loss, torch.zeros_like(loss_d))
        weight_s = torch.where(loss_s > loss, loss_s - loss, torch.zeros_like(loss_s))

        loss = loss + loss_d * weight_d + loss_s * weight_s
        return loss