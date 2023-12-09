import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(SentimentLSTM, self).__init__()

        # Embedding 층
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM 층
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )

        # Fully Connected 층
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        # Dropout 층
        self.dropout = nn.Dropout(dropout)


    def forward(self, text, text_lengths):
        # text: (batch_size, max_length)
        embedded = self.embedding(text)
        # embedded: (batch_size, max_length, embedding_dim)

        # 패딩된 시퀀스를 LSTM에 전달
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # LSTM 출력을 다시 패딩된 형태로 변환
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 마지막 타임스텝의 hidden state를 사용
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) if self.lstm.bidirectional else hidden[-1,:,:])
        
        # Fully Connected 층 적용
        output = self.fc(hidden)

        return output

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, nhead, num_encoder_layers, num_classes, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)  # Transformer 모델의 입력 형태로 변경

        # 분류 작업을 위해 `text`를 소스 및 타겟으로 사용한다고 가정합니다.
        # 만약 시퀀스 투 시퀀스 작업이라면 `tgt`를 다르게 처리해야 할 수 있습니다.
        transformer_output = self.transformer(src=embedded, tgt=embedded)

        # [CLS] 토큰의 표현만 가져오기 (첫 번째 토큰)
        pooled_output = transformer_output[0, :, :]

        # 대안으로 평균이나 최대 풀링과 같은 풀링 방법을 사용할 수 있습니다.
        # pooled_output = torch.mean(transformer_output, dim=0)
        # pooled_output = torch.max(transformer_output, dim=0).values

        output = self.fc(pooled_output)
        return output

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, nhead, num_layers, dropout=0.5):
        super(Seq2SeqTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src_embedded = self.embedding(src)
        src_embedded = src_embedded.permute(1, 0, 2)  # Transformer 모델의 입력 형태로 변경

        tgt_embedded = self.embedding(tgt)
        tgt_embedded = tgt_embedded.permute(1, 0, 2)  # Transformer 모델의 입력 형태로 변경

        transformer_encoder_output = self.transformer_encoder(src_embedded)
        transformer_decoder_output = self.transformer_decoder(tgt_embedded, transformer_encoder_output)

        # 최종 출력 (시퀀스의 각 위치에 대한 예측)
        output = self.fc(transformer_decoder_output)

        return output
