from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import quick_grad_check
import autograd.scipy.stats.norm as norm
from black_box_svi import black_box_variational_inference
from optimizers import adam


def sigmoid(x):
    return 0.5*(np.tanh(x) + 1)

def logsumexp(X, axis, keepdims=False):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=keepdims))

def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity):
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    N = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]
        '''
        for m, n in shapes:
            yield W_vect[:m*n].reshape((m,n)), W_vect[m*n:m*n+n]
            W_vect = W_vect[(m+1)*n:]
        '''
    def predictions(weights, inputs):
        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        #return outputs - logsumexp(outputs, axis=1, keepdims=True)
        return outputs


    def logprob(weights, inputs, targets):
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        #preds = sigmoid(np.mean(predictions(weights, inputs), axis=0))
        preds = predictions(weights, inputs)
        #log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_variance
        num_samples = preds.shape[0]
        exp_pred = np.exp(preds - np.max(preds,axis=2)[:,:,np.newaxis])
        normalized_pred = exp_pred / np.sum(exp_pred, axis=2)[:,:,np.newaxis]
        log_lik = np.sum(targets * np.log(normalized_pred + 1e-6), axis=(1,2))
        '''
        log_lik = np.zeros(num_samples)
        for mm in xrange(preds.shape[0]):
            pred_sample = preds[mm,:,:]
            exp_pred = np.exp(pred_sample - np.max(pred_sample, axis=1, keepdims=True))
            exp_pred /= exp_pred.sum()
            loglik_sample = log_prior[mm] - np.sum(targets * np.log(exp_pred + 1e-6))
            log_lik[mm] = loglik_sample
        '''
        return log_prior + log_lik

    '''
    def loss(weights, inputs, targets):
        eps = 1e-5
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        preds = sigmoid(predictions(weights, inputs))
        label_probabilities = targets[1,:] * np.log(preds + eps) + (1 - targets[1,:]) * np.log(1 - preds + eps)
        log_lik = -np.sum(label_probabilities, axis=(1,2))
        return log_prior + log_lik
    '''

    def frac_err(weights, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(np.mean(predictions(weights, X), axis=0), axis=1))

    return N, predictions, logprob, frac_err


def load_mnist():
    print("Loading training data...")
    import imp, urllib
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    source, _ = urllib.urlretrieve(
        'https://raw.githubusercontent.com/HIPS/Kayak/master/examples/data.py')
    data = imp.load_source('data', source).mnist()
    train_images, train_labels, test_images, test_labels = data
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def make_batches(N_data, batch_size):
    return [slice(i, min(i+batch_size, N_data))
            for i in range(0, N_data, batch_size)]



def extract_weights(variational_params, num_weights):
    mean, log_std = variational_params[:num_weights], variational_params[num_weights:]
    sample_weights = rs.randn(num_samples, num_weights) * np.exp(log_std) + mean
    #weights = np.mean(sample_weights, axis=0)
    return sample_weights



if __name__ == '__main__':
    # Network parameters
    layer_sizes = [784, 200, 100, 10]
    L2_reg = .01
    D=784

    # Training parameters
    param_scale = 0.1
    learning_rate = 1e-2
    momentum = 0.9
    batch_size = 100
    num_epochs = 50

    # Load and process MNIST data (borrowing from Kayak)
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    rbf = lambda x: norm.pdf(x, 0, 1)
    relu = lambda x: np.maximum(x, 0.0)

    # Make neural net functions
    num_weights, predictions, logprob, frac_err = \
        make_nn_funs(layer_sizes,
                     L2_reg,
                     noise_variance=0.01,
                     nonlinearity=relu)

    #loss_grad = grad(log_prob)

    # Initialize weights
    rs = npr.RandomState(0)
    num_samples = 20
    init_mean = rs.randn(num_weights)
    init_log_std = -3 * np.ones(num_weights)
    variational_params = np.concatenate([init_mean, init_log_std])

    def print_perf(epoch, W):
        test_perf  = frac_err(W, test_images, test_labels)
        #train_perf = frac_err(W, train_images, train_labels)
        print("Epoch %d | Test Error = %1.4f" % (epoch, test_perf))
        #print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

    #print("    Epoch      |    Train err  |   Test err  ")

    # Train with sgd
    batch_idxs = make_batches(train_images.shape[0], batch_size)
    cur_dir = np.zeros(num_weights*2)

    for epoch in range(num_epochs):
        for idxs in batch_idxs:
            log_posterior = lambda weights: logprob(weights, train_images[idxs], train_labels[idxs])

            objective, gradient, unpack_params = \
                black_box_variational_inference(log_posterior, num_weights, num_samples)

            grad_w = gradient(variational_params)
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_w
            variational_params -= learning_rate * cur_dir

            #variational_params = adam(gradient, variational_params, step_size=0.1, num_iters=10)

            weights = extract_weights(variational_params, num_weights)
            print_perf(epoch, weights)
