import torch
import torch.nn as nn
import torch.nn.functional as F

class XRAnomalyDetector(nn.Module):
    def __init__(self, input_dim=22, num_classes=4, cnn_out_channels=64, lstm_hidden=128):
        super(XRAnomalyDetector, self).__init__()
        
        self.stream1 = nn.Conv1d(
            in_channels=input_dim, out_channels=cnn_out_channels, 
            kernel_size=3, padding=1, dilation=1
        )
        self.stream2 = nn.Conv1d(
            in_channels=input_dim, out_channels=cnn_out_channels, 
            kernel_size=3, padding=2, dilation=2
        )
        
        combined_channels = cnn_out_channels * 2
        self.bn1 = nn.BatchNorm1d(combined_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv_fuse = nn.Conv1d(
            in_channels=combined_channels, out_channels=combined_channels, 
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(combined_channels)
        
        self.bilstm = nn.LSTM(
            input_size=combined_channels, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        self.attention = nn.Linear(lstm_hidden * 2, 1)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64) 
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        
        s1 = self.stream1(x) 
        s2 = self.stream2(x) 
        
        x = torch.cat((s1, s2), dim=1) 
        x = F.relu(self.bn1(x))
        x = self.pool(x) 
        x = F.relu(self.bn2(self.conv_fuse(x))) 
        
        x = x.permute(0, 2, 1) 
        
        lstm_out, _ = self.bilstm(x) 
        
        attn_scores = self.attention(lstm_out) 
        attn_weights = torch.softmax(attn_scores, dim=1) 
        context = (lstm_out * attn_weights).sum(dim=1) 
        
        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        logits = self.fc2(out) 
        
        return logits, attn_weights