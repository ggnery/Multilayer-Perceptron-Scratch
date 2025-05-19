from typing import Tuple, List

import gzip
import pickle
import numpy as np
import torch

class MNIST:
    training_data: Tuple[torch.Tensor, torch.Tensor | None]
    validation_data: Tuple[torch.Tensor, torch.Tensor]
    test_data: Tuple[torch.Tensor, torch.Tensor]
    device: torch.device
    
    def __init__(self,  device: torch.device):
        """Return the MNIST object that contais the training data,
        the validation data, and the test data.

        The ``training_data`` is returned as a tuple with two entries.
        The first entry contains the actual training images.  This is a
        numpy ndarray with 50,000 entries.  Each entry is, in turn, a
        numpy ndarray with 784 values, representing the 28 * 28 = 784
        pixels in a single MNIST image.

        The second entry in the ``training_data`` tuple is a numpy ndarray
        containing 50,000 entries.  Those entries are just the digit
        values (0...9) for the corresponding images contained in the first
        entry of the tuple.

        The ``validation_data`` and ``test_data`` are similar, except
        each contains only 10,000 images.

        This is a nice data format, but for use in neural networks it's
        helpful to modify the format of the ``training_data`` a little.
        That's done in the wrapper function ``format_data()``, see
        below.
        """
        f = gzip.open('./data/mnist.pkl.gz', 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        
        self.training_data = tuple([torch.from_numpy(x).to(device) for x in training_data])
        self.validation_data = tuple([torch.from_numpy(x).to(device) for x in validation_data])
        self.test_data = tuple([torch.from_numpy(x).to(device) for x in test_data])
        self.device = device
        
        f.close()

    def format_data(self):
        """Return a tuple containing ``(training_data, validation_data,
        test_data)``. Based on ``load_data``, but the format is more
        convenient for use in our implementation of neural networks.

        In particular, ``training_data`` is a list containing 50,000
        2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
        containing the input image.  ``y`` is a 10-dimensional
        numpy.ndarray representing the unit vector corresponding to the
        correct digit for ``x``.

        ``validation_data`` and ``test_data`` are lists containing 10,000
        2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
        numpy.ndarry containing the input image, and ``y`` is the
        corresponding classification, i.e., the digit values (integers)
        corresponding to ``x``.

        Obviously, this means we're using slightly different formats for
        the training data and the validation / test data.  These formats
        turn out to be the most convenient for use in our neural network
        code."""
        training_inputs = [torch.reshape(x, (784, 1)).squeeze(1) for x in self.training_data[0]]
        training_results = [MNIST.vectorized_result(y) for y in self.training_data[1]]
        training_data = zip(training_inputs, training_results)
        
        validation_inputs = [torch.reshape(x, (784, 1)).squeeze(1) for x in self.validation_data[0]]
        validation_data = zip(validation_inputs, self.validation_data[1])
        
        test_inputs = [torch.reshape(x, (784, 1)).squeeze(1) for x in self.test_data[0]]
        test_data = zip(test_inputs, self.test_data[1])
            
        return (training_data, validation_data, test_data)

    @staticmethod
    def vectorized_result(j: torch.Tensor):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = torch.zeros(10)
        e[j] = 1.0
        return e