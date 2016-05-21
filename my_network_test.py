__author__ = 'skao'

import unittest
import my_network as net
import network as check
import face_detection as fd

class MyTestCase(unittest.TestCase):


    def setUp(self):

        self.data = [[x,y] for x,y in zip(net.np.repeat(net.np.arange(-10,10,1),3).reshape(20,3), net.np.repeat([1,0],20).reshape(20,2))]
        self.nnet = net.My_Network([len(self.data[0][0]),5,len(self.data[0][1])])
        #self.anet = check.Network([len(self.data[0][0]),5,len(self.data[0][1])])
        self.data2 = fd.read_data('C:/Users/skao/PycharmProjects/DeepLearning/data/training/training.csv','C:/Users/skao/PycharmProjects/DeepLearning/data/testing/testing.csv',1500 )
        self.nnet2 = net.My_Network([len(self.data2[0][0]),100,500,50, len(self.data2[0][1])])

    def tearDown(self):
        self.nnet = None

    def test_case(self):

        self.nnet.SGD(self.data,1,5,1)
        results = self.nnet.evaluate(self.data[1:10])
        print(self.nnet2.cost(results[0], results[1]))
        print(results)
        self.nnet2.SGD(self.data2, 5, 500, 1)
        results2 = self.nnet2.evaluate(self.data2[1:20])
        print(results2)
        print(self.nnet2.cost(results2[0], results2[1]))

    def test_default(self):
        self.assertEqual(self.nnet.num_layers, 3)
        self.assertListEqual(self.nnet.sizes, [3,5,2])
        self.assertTupleEqual(self.nnet.biases[0].shape,(5,1))
        self.assertTupleEqual(self.nnet.biases[1].shape,(2,1))
        self.assertTupleEqual(self.nnet.weights[0].shape,(5,3))
        self.assertTupleEqual(self.nnet.weights[1].shape,(2,5))

    def test_sigmoid(self):
        self.assertEqual(self.nnet.sigmoid(0), 1/2)

    def test_feedforward(self):
        self.nnet.biases = [net.np.zeros((y,1)) for y in [5,2]]
        self.nnet.weights = [net.np.ones((y,x)) for x,y in [(3,5),(5,2)]]

        self.assertItemsEqual(self.nnet.feedforward(net.np.zeros((3,1)))[2],net.np.repeat(net.sigmoid(2.5),2))


    def test_backprop(self):

        x_data = [x for x,y in self.data[10:11]]
        y_data = [y for x,y in self.data[10:11]]
        self.nnet.biases = [net.np.zeros((y,1)) for y in [5,2]]
        self.nnet.weights = [net.np.zeros((y,x)) for x,y in [(3,5),(5,2)]]

        (we, bi) = self.nnet.backprop(x_data,y_data)
        self.assertTrue(we[0].__eq__(net.np.zeros((5,3))).all())
        self.assertTrue(we[1].__eq__(net.np.repeat(0.0625,10).reshape(2,5)).all())
        self.assertTrue(bi[0].__eq__(net.np.zeros((5,1))).all())
        self.assertTrue(bi[1].__eq__(net.np.repeat(0.125,2).reshape(2,1)).all())

    def test_update_mini_batch(self):

        mini_batch = [self.data[10:11]]
        self.nnet.biases = [net.np.zeros((y,1)) for y in [5,2]]
        self.nnet.weights = [net.np.zeros((y,x)) for x,y in [(3,5),(5,2)]]
        self.nnet.update_mini_batch(mini_batch, 1)
        self.assertTrue(self.nnet.weights[0].__eq__(net.np.zeros((5,3))).all())
        self.assertTrue(self.nnet.weights[1].__eq__(net.np.repeat(-0.0625,10).reshape(2,5)).all())
        self.assertTrue(self.nnet.biases[0].__eq__(net.np.zeros((5,1))).all())
        self.assertTrue(self.nnet.biases[1].__eq__(net.np.repeat(-0.125,2).reshape(2,1)).all())

    def test_SGD(self):
        self.nnet.biases = [net.np.zeros((y,1)) for y in [5,2]]
        self.nnet.weights = [net.np.zeros((y,x)) for x,y in [(3,5),(5,2)]]
        #self.anet.biases = [net.np.zeros((y,1)) for y in [5,2]]
        #self.anet.weights = [net.np.zeros((y,x)) for x,y in [(3,5),(5,2)]]
        self.nnet.SGD(self.data[10:11], 1, 1, 1)
        #self.anet.SGD(self.data[10:11], 1, 1, 1)
        self.assertTrue(self.nnet.weights[0].__eq__(net.np.zeros((5,3))).all())
        self.assertTrue(self.nnet.weights[1].__eq__(net.np.repeat(-0.0625,10).reshape(2,5)).all())
        self.assertTrue(self.nnet.biases[0].__eq__(net.np.zeros((5,1))).all())
        self.assertTrue(self.nnet.biases[1].__eq__(net.np.repeat(-0.125,2).reshape(2,1)).all())
        #self.assertTrue(self.anet.weights[0].__eq__(net.np.zeros((5,3))).all())
        #self.assertTrue(self.anet.weights[1].__eq__(net.np.repeat(-0.0625,10).reshape(2,5)).all())
        #self.assertTrue(self.anet.biases[0].__eq__(net.np.zeros((5,1))).all())
        #self.assertTrue(self.anet.biases[1].__eq__(net.np.repeat(-0.125,2).reshape(2,1)).all())#self.assertTrue(self.nnet.SGD(self.data, 1, 5, 1), )

    def test_sigmoid(self):
        self.assertTrue(net.sigmoid(0), 0.5)


    def test_sigmoid_prime(self):
        self.assertTrue(net.sigmoid_prime(0), 0.25)



if __name__ == '__main__':
    unittest.main()

