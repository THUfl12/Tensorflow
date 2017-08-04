Use Keras and Tensorflow to develop a simple distributed example.
There is one parameter server(ps), and one worker server.
This model is a complicated one, you can just replace it for a easier one.

![ps.py](https://github.com/THUfl12/Tensorflow/blob/master/Keras/Simple_Distributed_Test/ps.py): start parameter server. <br>
![worker.py](https://github.com/THUfl12/Tensorflow/blob/master/Keras/Simple_Distributed_Test/worker.py): start worker server 0 which is running on 172.22.191.46:2226. <br>
![worker_1.py](https://github.com/THUfl12/Tensorflow/blob/master/Keras/Simple_Distributed_Test/worker_1.py): start worker server 1, which is running on 172.22.191.46:2225. <br>
Be sure the ip address is right!

Reference:
------------------------------------------------
[Tensorflow.org: Distributed TensorFlow ](http://www.tensorflow.org/deploy/distributed) <br>
[Imanol Schlag: Distributed Tensorflow Example ](http://ischlag.github.io/2016/06/12/async-distributed-tensorflow/) <br>
[fchollet: keras_distributed ](http://gist.github.com/fchollet/2c9b029f505d94e6b8cd7f8a5e244a4e)
