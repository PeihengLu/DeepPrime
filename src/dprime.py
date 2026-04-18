# Data for normalization should be in models directory

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from glob import glob
from deepprime.src.utils import seq_concat, select_cols

import lightning.pytorch as pl  # type: ignore[reportMissingImports]



class GeneInteractionModel(nn.Module):

    def __init__(self, hidden_size, num_layers, num_features=24, dropout=0.1):
        super(GeneInteractionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=128, kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.r = nn.GRU(128, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.s = nn.Linear(2 * hidden_size, 12, bias=False)

        self.d = nn.Sequential(
            nn.Linear(num_features, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128, bias=False)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(140),
            nn.Dropout(dropout),
            nn.Linear(140, 1, bias=True),
        )

    def forward(self, g, x):
        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g)
        g, _ = self.r(torch.transpose(g, 1, 2))
        g = self.s(g[:, -1, :])

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)


class GeneInteractionLightningModule(pl.LightningModule):
    """Lightning wrapper for DeepPrime GeneInteractionModel."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.criterion = nn.MSELoss()

    def training_step(self, batch, _batch_idx):
        g, x, y = batch
        pred = self.model(g.permute((0, 3, 1, 2)), x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, _batch_idx):
        g, x, y = batch
        pred = self.model(g.permute((0, 3, 1, 2)), x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )



def calculate_deepprime_score(df_input, pe_system='PE2max', cell_type='HEK293T'):

    os.environ['CUDA_VISIBLE_DEVICES']='0'
    if torch.cuda.is_available(): device = 'cuda'
    else : device = 'cpu'
    
    from deepprime.models.load_model import load_deepprime

    model_dir, model_type = load_deepprime(pe_system, cell_type, silent=True)

    mean = pd.read_csv('%s/DeepPrime_base/mean.csv' % model_dir, header=None, index_col=0).squeeze()
    std  = pd.read_csv('%s/DeepPrime_base/std.csv' % model_dir, header=None, index_col=0).squeeze()

    test_features = select_cols(df_input)

    g_test = seq_concat(df_input)
    x_test = (test_features - mean) / std

    g_test = torch.tensor(g_test, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32, device=device)

    models = [m_files for m_files in glob('%s/%s/*.pt' % (model_dir, model_type))]
    preds  = []

    for m in models:
        model = GeneInteractionModel(hidden_size=128, num_layers=1).to(device)
        model.load_state_dict(torch.load(m, map_location=torch.device(device)))
        model.eval()
        with torch.no_grad():
            g, x = g_test, x_test
            g = g.permute((0, 3, 1, 2))
            pred = model(g, x).detach().cpu().numpy()
        preds.append(pred)
    
    # AVERAGE PREDICTIONS
    preds = np.squeeze(np.array(preds))
    preds = np.mean(preds, axis=0)
    preds = np.exp(preds) - 1

    return preds

# # SAVE RESULTS

# preds = pd.DataFrame(preds, columns=['Predicted_PE_efficiency'])
# preds.to_csv('prediction.csv', index=False)

