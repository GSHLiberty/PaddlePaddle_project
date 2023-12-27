import paddle
"""
RNN with configurable architecture.
"""


class RNN(paddle.nn.Layer):
    """
    General Recurrent Neural Network module.
    Input: tensor of shape (seq_len, batch, input_size)
    Output: tensor of shape (seq_len, batch, hidden_size * num_directions)
    """

    def __init__(self, input_size, hidden_size, output_projection_size=None,
        num_layers=1, bidirectional=True, cell_type='lstm', dropout=0, pack
        =False, batch_first=False, init_method='default'):
        super().__init__()
        self.input_layer = paddle.nn.Linear(in_features=input_size,
            out_features=hidden_size)
        if output_projection_size is not None:
            self.output_layer = paddle.nn.Linear(in_features=hidden_size * 
                2 if bidirectional else hidden_size, out_features=
                output_projection_size)
        self.pack = pack
        network = self._get_rnn(cell_type)
        self.network = network(input_size=input_size, hidden_size=
            hidden_size, num_layers=num_layers, bidirectional=bidirectional,
            dropout=dropout, batch_first=batch_first)

    def forward(self, input_variable):
        outputs, hidden = self.network(input_variable)
        if self.pack:
            outputs = self.output_layer(outputs)
# >>>            padded_outputs, lengths = torch.nn.utils.rnn.pad_packed_sequence(
#                 outputs)
#             if hasattr(self, 'output_layer'):
# >>>                outputs = torch.nn.utils.rnn.pack_padded_sequence(self.
#                     output_layer(padded_outputs), lengths)
        elif hasattr(self, 'output_layer'):
            outputs = self.output_layer(outputs)
        return outputs, hidden

    def _get_rnn(self, rnn_type):
        rnn_type = rnn_type.lower()
        if rnn_type == 'gru':
            network = paddle.nn.GRU
        elif rnn_type == 'lstm':
            network = paddle.nn.LSTM
        else:
            raise ValueError('Invalid RNN type %s' % rnn_type)
        return network
