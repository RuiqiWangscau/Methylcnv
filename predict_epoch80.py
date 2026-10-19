#!/usr/bin/env python
"""Run a saved legacy CNN-LSTM on already-preprocessed sequence arrays. No training."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/public/home/wangruiqi/bam/cnv_benchmark/comparison_with_coordinates/best_model_epoch_80.pth'))
    parser.add_argument('--input', type=Path, required=True, help='NPZ containing original preprocessed 3-D sequences; raw bin-feature NPZ is not supported')
    parser.add_argument('--input_key', default='X_test', help='Sequence array key in the NPZ')
    parser.add_argument('--labels_key', default='y_test', help='Optional 1-D sequence-label array key')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--class_names', nargs=3, default=None, help='Verified class names in class-index order; omitted by default because weights do not record the mapping')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('Batch size and thread count must be positive')
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(args.threads)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model = CNNLSTM(input_dim=29, num_classes=3)
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    with np.load(args.input, allow_pickle=False) as data:
        if args.input_key not in data:
            raise KeyError(f'Missing sequence key {args.input_key}; available keys: {data.files}')
        x = np.asarray(data[args.input_key], dtype=np.float32)
        labels = np.asarray(data[args.labels_key]) if args.labels_key in data else None
    if x.ndim != 3 or x.shape[2] != 29 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f'Expected original preprocessed sequences with shape (N, sequence_length, 29), got {x.shape}. Do not reshape raw features to bypass this check.')
    if not np.isfinite(x).all():
        raise ValueError('Input contains non-finite values')
    if labels is not None and (labels.shape != (len(x),) or not np.isin(labels, [0, 1, 2]).all()):
        raise ValueError('Optional labels must be one sequence label per row, encoded 0/1/2')
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(x[start:start+args.batch_size]).to(device)
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    predictions = probabilities.argmax(axis=1)
    table = pd.DataFrame({'sequence_id': np.arange(len(x)), 'prediction_id': predictions,
                          'prob_class0': probabilities[:, 0], 'prob_class1': probabilities[:, 1],
                          'prob_class2': probabilities[:, 2]})
    payload = {'predictions': predictions, 'probabilities': probabilities}
    if labels is not None:
        table['true_label_id'] = labels.astype(np.int64)
        payload['labels'] = labels.astype(np.int64)
    if args.class_names:
        names = np.asarray(args.class_names)
        table['prediction'] = names[predictions]
        if labels is not None:
            table['true_label'] = names[labels.astype(int)]
    args.output.mkdir(parents=True)
    table.to_csv(args.output/'predictions.tsv', sep='\t', index=False)
    np.savez_compressed(args.output/'predictions.npz', **payload)
    (args.output/'input_note.txt').write_text(
        'Applied the saved legacy CNN-LSTM without training or scaler fitting. Input must use the original feature order, cleaning, scaling, sequence length and window construction. Architecture match does not prove training provenance. Evaluation on training data is not independent.\n', encoding='utf-8')
    print(f'Completed prediction without training: {len(x)} sequences -> {args.output}')

class CNNLSTM(nn.Module):

    def __init__(self, input_dim, num_classes=3):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, 48, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, 48, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(input_dim, 48, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(192)
        self.dropout_cnn = nn.Dropout(0.25)
        self.lstm1 = nn.LSTM(192, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.res_linear = nn.Linear(input_dim, 64)
        self.fc1 = nn.Linear(128 + 64, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x_res = x[:, -1, :].clone()
        x_t = x.transpose(1, 2)
        c1 = self.relu(self.conv1(x_t))
        c2 = self.relu(self.conv2(x_t))
        c3 = self.relu(self.conv3(x_t))
        c4 = self.relu(self.conv4(x_t))
        cnn_out = torch.cat([c1, c2, c3, c4], dim=1)
        cnn_out = self.batch_norm1(cnn_out)
        cnn_out = self.dropout_cnn(cnn_out)
        lstm_in = cnn_out.transpose(1, 2)
        lstm1_out, _ = self.lstm1(lstm_in)
        attn_out, _ = self.attention(lstm1_out, lstm1_out, lstm1_out)
        attn_out = attn_out + lstm1_out
        lstm2_out, _ = self.lstm2(attn_out)
        lstm_last = lstm2_out[:, -1, :]
        lstm_norm = self.batch_norm3(lstm_last)
        lstm_drop = self.dropout2(lstm_norm)
        res_out = self.relu(self.res_linear(x_res))
        combined = torch.cat([lstm_drop, res_out], dim=1)
        fc1_out = self.relu(self.fc1(combined))
        fc1_drop = self.dropout3(fc1_out)
        output = self.fc2(fc1_drop)
        return output



if __name__ == "__main__":
    main()
