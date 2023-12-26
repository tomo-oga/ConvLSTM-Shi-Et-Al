# An Implementation of the Original Approach of ConvLSTM by Shi _et. al._

## Summary
A fully functional and explicitely followed implementation from <i>Shi et. al</i>'s work on ["Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting."](https://arxiv.org/abs/1506.04214) 

**Note:** While this **`nn.Module`** is fully functional, it's parameter count is quite high from repeated convolutional matrices, that could be simplified. This original approach shown in **`conv_lstm_shi-et-al.py`** is meant to be an explicitely followed implementation of that work. 
`conv_lstm_efficient.py` is the more efficient and practical implementation of `ConvLSTMCell`. 

## Models
* `conv_lstm_shi-et-al.py` contains the original implementation for a `ConvLSTMCell`
* `conv_lstm_efficient.py` contains the efficient one convolultional layer imlementation of `ConvLSTMCell`
* `ConvLSTMLayer.py` contains an implementation for a `ConvLSTMLayer` to create reccurency and processing of temporal data for a `ConvLSTMCell`

## Example Implementation of `ConvLSTMLayer`
`layer = ConvLSTMLayer(64,64)`
## Dependencies:
`torch`
