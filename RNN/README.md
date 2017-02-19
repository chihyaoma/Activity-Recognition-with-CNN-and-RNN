### RNN for video prediction
We use the [RNN library](https://github.com/Element-Research/rnn) provided by Element-Research. This *main.lua* takes computed feature vectors from severla equally sampled video frames, and learns a generic RNN model to represent the videos aross all training video frames.

Using command line arguments or modify the code in *main.lua*, you can essentially modify the Learning rate, batch size, CUDA options, LSTM or GRU, dropout layers, and the dimension of hidden layers.


#### There are several things you can directly try

- use vanilla RNN, LSTM, or GRU
- optimizer: SGD, ADAM, RMSPROP
- number of segments used from the video
- temporal pooling methods


#### Contact: [Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-m/me) at <cyma@gatech.edu> or [[LinkedIn]](https://www.linkedin.com/in/chih-yao-ma-9b5b3063)
