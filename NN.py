import network3
from network3 import Network, ReLU
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from sklearn.externals import joblib

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 1178, 617),
                      filter_shape=(20, 1, 55, 6),
                      poolsize=(2, 2)),
        SoftmaxLayer(n_in=562*306, n_out=3)], mini_batch_size)
net.SGD(training_data, 90, mini_batch_size, 0.1,
        validation_data, test_data)
# net = joblib.load('./logData/net')
# print net.params[0].get_value()