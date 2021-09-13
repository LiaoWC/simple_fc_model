# Import
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import graphviz
from typing import Optional, Tuple


# Loss function: MSE

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [], []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        labels.append(0 if pt[0] > pt[1] else 1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_xor_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def generate_xor_easy_more(k=1):
    inputs = []
    labels = []
    for i in range(11 * k):
        inputs.append([0.1 * i / k, 0.1 * i / k])
        labels.append(0)
        if 0.1 * i == 0.5 * k:
            continue
        inputs.append([0.1 * i / k, 1 - 0.1 * i / k])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(-1, 1)


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)
    for i in range(x.shape[0]):
        plt.plot(x[i][0], x[i][1], 'ro' if y[i] == 0 else 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predict Result', fontsize=18)
    for i in range(x.shape[0]):
        plt.plot(x[i][0], x[i][1], 'ro' if pred_y[i] == 0 else 'bo')
    plt.show()


class Model:
    def __init__(self, fc1feat: int, fc2feat: int):
        self.infeat, self.fc1feat, self.fc2feat, self.fc3feat = 2, fc1feat, fc2feat, 1
        # Init weights (PyTorch way)
        # Not support bias
        self.fc1w = np.random.uniform(-np.sqrt((1. / self.infeat)), np.sqrt((1. / self.infeat)),
                                      (self.fc1feat, self.infeat))
        self.fc2w = np.random.uniform(-np.sqrt((1. / self.fc1feat)), np.sqrt((1. / self.fc1feat)),
                                      (self.fc2feat, self.fc1feat))
        self.fc3w = np.random.uniform(-np.sqrt((1. / self.fc2feat)), np.sqrt((1. / self.fc2feat)),
                                      (self.fc3feat, self.fc2feat))
        # Temporary calculation result
        self.inp = np.zeros((self.infeat, 1))
        self.fc1sum = np.zeros((self.fc1feat, 1))
        self.fc2sum = np.zeros((self.fc2feat, 1))
        self.fc3sum = np.zeros((self.fc3feat, 1))
        # Gradient
        self.sum_fc1d, self.sum_fc2d, self.sum_fc3d = None, None, None

    @staticmethod
    def plot_model(model: "Model"):
        d = graphviz.Digraph(filename='NN')
        # Node index from 1
        # Nodes
        with d.subgraph() as s:
            s.attr(rank='same')
            for i in range(1, model.infeat + 1):
                s.node('input_' + str(i))
        with d.subgraph() as s:
            s.attr(rank='same')
            for i in range(1, model.fc1feat + 1):
                s.node('fc1_' + str(i), color='lightblue2', style='filled')
        with d.subgraph() as s:
            s.attr(rank='same')
            for i in range(1, model.fc1feat + 1):
                s.node('sig1_' + str(i), shape='diamond', style='filled', color='lightgrey')
        with d.subgraph() as s:
            s.attr(rank='same')
            for i in range(1, model.fc2feat + 1):
                s.node('fc2_' + str(i), color='lightblue2', style='filled')
        with d.subgraph() as s:
            s.attr(rank='same')
            for i in range(1, model.fc2feat + 1):
                s.node('sig2_' + str(i), shape='diamond', style='filled', color='lightgrey')
        with d.subgraph() as s:
            s.attr(rank='same')
            for i in range(1, model.fc3feat + 1):
                s.node('out_' + str(i), color='orange', style='filled')
        with d.subgraph() as s:
            s.attr(rank='same')
            for i in range(1, model.fc3feat + 1):
                s.node('sig_out_' + str(i), shape='diamond', style='filled', color='lightgrey')
        # Edges
        for i in range(1, model.infeat + 1):
            for j in range(1, model.fc1feat + 1):
                d.edge('input_' + str(i), 'fc1_' + str(j), label=f'{model.fc1w[j - 1][i - 1]:.4f}', len='10.')
        for i in range(1, model.fc1feat + 1):
            d.edge('fc1_' + str(i), 'sig1_' + str(i), label='sigmoid')
        for i in range(1, model.fc1feat + 1):
            for j in range(1, model.fc2feat + 1):
                d.edge('sig1_' + str(i), 'fc2_' + str(j), label=f'{model.fc2w[j - 1][i - 1]:.4f}', len='500.')
        for i in range(1, model.fc2feat + 1):
            d.edge('fc2_' + str(i), 'sig2_' + str(i), label='sigmoid')
        for i in range(1, model.fc2feat + 1):
            for j in range(1, model.fc3feat + 1):
                d.edge('sig2_' + str(i), 'out_' + str(j), label=f'{model.fc3w[j - 1][i - 1]:.4f}')
        for i in range(1, model.fc3feat + 1):
            d.edge('out_' + str(i), 'sig_out_' + str(i), label='sigmoid')
        #
        d.attr(overlap='true')
        d.view()

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def der_sigmoid(x):  # derivative
        x = Model.sigmoid(x)
        return x * (1 - x)

    def _init_grad(self):
        self.sum_fc1d = np.zeros((self.fc1feat, self.infeat))
        self.sum_fc2d = np.zeros((self.fc2feat, self.fc1feat))
        self.sum_fc3d = np.zeros((self.fc3feat, self.fc2feat))

    def _init_weights(self):
        self.fc1w = np.random.uniform(-np.sqrt((1. / self.infeat)), np.sqrt((1. / self.infeat)),
                                      (self.fc1feat, self.infeat))
        self.fc2w = np.random.uniform(-np.sqrt((1. / self.fc1feat)), np.sqrt((1. / self.fc1feat)),
                                      (self.fc2feat, self.fc1feat))
        self.fc3w = np.random.uniform(-np.sqrt((1. / self.fc2feat)), np.sqrt((1. / self.fc2feat)),
                                      (self.fc3feat, self.fc2feat))

    def _forward(self, x_input: np.array):
        # Check input dimension
        if x_input.shape == (self.infeat,):
            x = np.expand_dims(x_input, axis=1)
        else:
            x = x_input
        if x.shape != (self.infeat, 1):
            raise ValueError(f"Input dim must be ({self.infeat},) or ({self.infeat}, 1) but got {x_input.shape}")
        self.inp = deepcopy(x)
        # Forward
        x = np.matmul(self.fc1w, x)
        self.fc1sum = deepcopy(x)
        x = self.sigmoid(x)
        x = np.matmul(self.fc2w, x)
        self.fc2sum = deepcopy(x)
        x = self.sigmoid(x)
        x = np.matmul(self.fc3w, x)
        self.fc3sum = deepcopy(x)
        x = self.sigmoid(x)
        return x  # 2 dims - 1x1

    def no_act_forward(self, x_input: np.array):
        # Check input dimension
        if x_input.shape == (self.infeat,):
            x = np.expand_dims(x_input, axis=1)
        else:
            x = x_input
        if x.shape != (self.infeat, 1):
            raise ValueError(f"Input dim must be ({self.infeat},) or ({self.infeat}, 1) but got {x_input.shape}")
        self.inp = deepcopy(x)
        # Forward
        x = np.matmul(self.fc1w, x)
        self.fc1sum = deepcopy(x)
        x = np.matmul(self.fc2w, x)
        self.fc2sum = deepcopy(x)
        x = np.matmul(self.fc3w, x)
        self.fc3sum = deepcopy(x)
        x = self.sigmoid(x)
        return x  # 2 dims - 1x1

    def _backward(self, y_true: float, y_pred: float):
        #
        outd = -2 * (y_true - y_pred)
        fc3sumd = (outd * self.der_sigmoid(self.fc3sum[0][0])).reshape((1, 1))
        fc3d = np.matmul(fc3sumd, self.sigmoid(self.fc2sum).T)

        fc2sumd = np.matmul(self.fc3w.T, fc3sumd) * self.der_sigmoid(self.fc2sum)

        fc2d = np.matmul(fc2sumd, self.sigmoid(self.fc1sum).T)

        fc1sumd = np.matmul(self.fc2w.T, fc2sumd) * self.der_sigmoid(self.fc1sum)

        fc1d = np.matmul(fc1sumd, self.inp.T)
        #
        self.sum_fc1d += fc1d
        self.sum_fc2d += fc2d
        self.sum_fc3d += fc3d

    def no_act_backward(self, y_true: float, y_pred: float):
        #
        outd = -2 * (y_true - y_pred)
        fc3sumd = (outd * self.fc3sum[0][0]).reshape((1, 1))
        fc3d = np.matmul(fc3sumd, self.fc2sum.T)

        fc2sumd = np.matmul(self.fc3w.T, fc3sumd) * self.fc2sum
        # fc2sumd = (fc3d.T * self.der_sigmoid(self.fc2sum))

        fc2d = np.matmul(fc2sumd, self.fc1sum.T)

        fc1sumd = np.matmul(self.fc2w.T, fc2sumd) * self.fc1sum
        # fc1sumd = np.expand_dims((fc2d.T * self.der_sigmoid(self.fc1sum)).sum(axis=1), axis=1)

        fc1d = np.matmul(fc1sumd, self.inp.T)
        #
        self.sum_fc1d += fc1d
        self.sum_fc2d += fc2d
        self.sum_fc3d += fc3d

    def train(self, x: np.ndarray, y: np.ndarray, batch_size_ratio: float, epochs: int, lr: float,
              early_start_acc: Optional[Tuple[float, int]] = None, display_freq: int = 1, plot: bool = False,
              plot_but_not_show: bool = False):
        if batch_size_ratio > 1. or batch_size_ratio <= 0:
            raise ValueError(f'Batch size must be <= 1 and > 0 but got {batch_size_ratio}')

        n_samples = x.shape[0]
        batch_size = int(n_samples * batch_size_ratio)
        n_batches = int(np.ceil(n_samples / batch_size))
        last_batch_size = n_samples - ((n_batches - 1) * batch_size)

        n_reach_early_stop_acc = 0
        n_accumulated_epochs = 0

        all_loss = []
        all_acc = []

        for epoch in range(1, epochs + 1):
            n_accumulated_epochs += 1
            if n_accumulated_epochs >= display_freq:
                print(f'=== epoch {epoch} === ', end='')
            indices = np.random.choice(n_samples, n_samples, replace=False)
            for batch in range(1, n_batches + 1):
                self._init_grad()
                if batch == n_batches:
                    cur_batch_size = deepcopy(last_batch_size)
                else:
                    cur_batch_size = deepcopy(batch_size)
                start_idx = (batch - 1) * batch_size
                end_idx = start_idx + cur_batch_size
                batch_x, batch_y = x[indices[start_idx:end_idx]], y[indices[start_idx:end_idx]]
                #
                for one_x, one_y in zip(batch_x, batch_y):
                    one_pred = self._forward(one_x)
                    self._backward(y_true=one_y, y_pred=one_pred)
                # Update weights
                self.fc1w -= lr * self.sum_fc1d / cur_batch_size
                self.fc2w -= lr * self.sum_fc2d / cur_batch_size
                self.fc3w -= lr * self.sum_fc3d / cur_batch_size
            # Evaluate
            loss = 0.0
            correct = 0
            for one_x, one_y in zip(x, y):  # one_y is 1-dim; one_pred is 2-dim
                one_pred = self._forward(one_x)
                loss += np.power(one_y[0] - one_pred[0][0], 2)
                pred_class = 1. if one_pred[0][0] >= 0.5 else 0.
                if pred_class == one_y[0]:
                    correct += 1
            loss /= n_samples
            acc = correct / n_samples
            all_loss.append(loss)
            all_acc.append(acc)
            if n_accumulated_epochs >= display_freq:
                print(f'Loss: {loss:.6f} ', end='')
                print(f'Training Accuracy: {acc:.6f}')
                n_accumulated_epochs = 0
            if early_start_acc:
                if acc > early_start_acc[0]:
                    n_reach_early_stop_acc += 1
                    if n_reach_early_stop_acc > early_start_acc[1]:
                        print(f'> {early_start_acc} accuracy early stop')
                        break
                else:
                    n_reach_early_stop_acc = 0
        if plot:
            plt.plot((np.arange(len(all_loss)) + 1), all_loss, label=f'Training Loss ({self.fc1feat}x{self.fc2feat})')
            plt.plot((np.arange(len(all_acc)) + 1), all_acc, label=f'Training Accuracy ({self.fc1feat}x{self.fc2feat})')
            plt.title('Training Curve')
            plt.xlabel('Epochs')
            plt.legend()
            if not plot_but_not_show:
                plt.show()

    def no_act_train(self, x: np.ndarray, y: np.ndarray, batch_size_ratio: float, epochs: int, lr: float,
                     early_start_acc: Optional[Tuple[float, int]] = None, display_freq: int = 1, plot: bool = False,
                     plot_but_not_show: bool = False):
        if batch_size_ratio > 1. or batch_size_ratio <= 0:
            raise ValueError(f'Batch size must be <= 1 and > 0 but got {batch_size_ratio}')

        n_samples = x.shape[0]
        batch_size = int(n_samples * batch_size_ratio)
        n_batches = int(np.ceil(n_samples / batch_size))
        last_batch_size = n_samples - ((n_batches - 1) * batch_size)

        n_reach_early_stop_acc = 0
        n_accumulated_epochs = 0

        all_loss = []
        all_acc = []

        for epoch in range(1, epochs + 1):
            n_accumulated_epochs += 1
            if n_accumulated_epochs >= display_freq:
                print(f'=== epoch {epoch} === ', end='')
            indices = np.random.choice(n_samples, n_samples, replace=False)
            for batch in range(1, n_batches + 1):
                self._init_grad()
                if batch == n_batches:
                    cur_batch_size = deepcopy(last_batch_size)
                else:
                    cur_batch_size = deepcopy(batch_size)
                start_idx = (batch - 1) * batch_size
                end_idx = start_idx + cur_batch_size
                batch_x, batch_y = x[indices[start_idx:end_idx]], y[indices[start_idx:end_idx]]
                #
                for one_x, one_y in zip(batch_x, batch_y):
                    one_pred = self.no_act_forward(one_x)
                    self.no_act_backward(y_true=one_y, y_pred=one_pred)
                # Update weights
                self.fc1w -= lr * self.sum_fc1d / cur_batch_size
                self.fc2w -= lr * self.sum_fc2d / cur_batch_size
                self.fc3w -= lr * self.sum_fc3d / cur_batch_size
            # Evaluate
            loss = 0.0
            correct = 0
            for one_x, one_y in zip(x, y):  # one_y is 1-dim; one_pred is 2-dim
                one_pred = self._forward(one_x)
                loss += np.power(one_y[0] - one_pred[0][0], 2)
                pred_class = 1. if one_pred[0][0] >= 0.5 else 0.
                if pred_class == one_y[0]:
                    correct += 1
            loss /= n_samples
            acc = correct / n_samples
            all_loss.append(loss)
            all_acc.append(acc)
            if n_accumulated_epochs >= display_freq:
                print(f'Loss: {loss:.6f} ', end='')
                print(f'Training Accuracy: {acc:.6f}')
                n_accumulated_epochs = 0
            if early_start_acc:
                if acc > early_start_acc[0]:
                    n_reach_early_stop_acc += 1
                    if n_reach_early_stop_acc > early_start_acc[1]:
                        print(f'> {early_start_acc} accuracy early stop')
                        break
                else:
                    n_reach_early_stop_acc = 0
        if plot:
            plt.plot((np.arange(len(all_loss)) + 1), all_loss, label=f'Training Loss ({self.fc1feat}x{self.fc2feat})')
            plt.plot((np.arange(len(all_acc)) + 1), all_acc, label=f'Training Accuracy ({self.fc1feat}x{self.fc2feat})')
            plt.title('Training Curve')
            plt.xlabel('Epochs')
            plt.legend()
            if not plot_but_not_show:
                plt.show()

    def test(self, test_x: np.ndarray, test_y: np.ndarray, print_loss: bool = True, print_acc: bool = True,
             print_pred_values: bool = False, return_pred_classes: bool = False):
        n_samples = test_x.shape[0]
        loss = 0.0
        correct = 0
        preds = []
        pred_classes = []
        for one_x, one_y in zip(test_x, test_y):  # one_y is 1-dim; one_pred is 2-dim
            one_pred = self._forward(one_x)
            preds.append(one_pred[0])
            loss += np.power(one_y[0] - one_pred[0][0], 2)
            pred_class = 1. if one_pred[0][0] >= 0.5 else 0.
            pred_classes.append(pred_class)
            if pred_class == one_y[0]:
                correct += 1
        loss /= n_samples
        preds = np.array(preds)
        if print_loss:
            print(f'Testing Loss: {loss:.6f}')
        if print_acc:
            print(f'Testing Accuracy: {correct / n_samples:.6f}')
        if print_pred_values:
            print(preds)
        if return_pred_classes:
            return pred_classes

    def train_diff_batch_size(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float,
                              early_start_acc: Optional[Tuple[float, int]] = None, display_freq: int = 1,
                              plot: bool = False,
                              batch_size_ratio_list: list = None):
        for batch_size_ratio in batch_size_ratio_list:
            self._init_weights()

            if batch_size_ratio > 1. or batch_size_ratio <= 0:
                raise ValueError(f'Batch size must be <= 1 and > 0 but got {batch_size_ratio}')

            n_samples = x.shape[0]
            batch_size = int(n_samples * batch_size_ratio)
            n_batches = int(np.ceil(n_samples / batch_size))
            last_batch_size = n_samples - ((n_batches - 1) * batch_size)

            n_reach_early_stop_acc = 0
            n_accumulated_epochs = 0

            all_loss = []
            all_acc = []

            for epoch in range(1, epochs + 1):
                n_accumulated_epochs += 1
                if n_accumulated_epochs >= display_freq:
                    print(f'=== epoch {epoch} (bs_ration={batch_size_ratio}) === ', end='')
                indices = np.random.choice(n_samples, n_samples, replace=False)
                for batch in range(1, n_batches + 1):
                    self._init_grad()
                    if batch == n_batches:
                        cur_batch_size = deepcopy(last_batch_size)
                    else:
                        cur_batch_size = deepcopy(batch_size)
                    start_idx = (batch - 1) * batch_size
                    end_idx = start_idx + cur_batch_size
                    batch_x, batch_y = x[indices[start_idx:end_idx]], y[indices[start_idx:end_idx]]
                    #
                    for one_x, one_y in zip(batch_x, batch_y):
                        one_pred = self._forward(one_x)
                        self._backward(y_true=one_y, y_pred=one_pred)
                    # Update weights
                    self.fc1w -= lr * self.sum_fc1d / cur_batch_size
                    self.fc2w -= lr * self.sum_fc2d / cur_batch_size
                    self.fc3w -= lr * self.sum_fc3d / cur_batch_size
                # Evaluate
                loss = 0.0
                correct = 0
                for one_x, one_y in zip(x, y):  # one_y is 1-dim; one_pred is 2-dim
                    one_pred = self._forward(one_x)
                    loss += np.power(one_y[0] - one_pred[0][0], 2)
                    pred_class = 1. if one_pred[0][0] >= 0.5 else 0.
                    if pred_class == one_y[0]:
                        correct += 1
                loss /= n_samples
                acc = correct / n_samples
                all_loss.append(loss)
                all_acc.append(acc)
                if n_accumulated_epochs >= display_freq:
                    print(f'Loss: {loss:.6f} ', end='')
                    print(f'Training Accuracy: {acc:.6f}')
                    n_accumulated_epochs = 0
                if early_start_acc:
                    if acc > early_start_acc[0]:
                        n_reach_early_stop_acc += 1
                        if n_reach_early_stop_acc > early_start_acc[1]:
                            print(f'> {early_start_acc} accuracy early stop')
                            break
                    else:
                        n_reach_early_stop_acc = 0
            if plot:
                plt.plot((np.arange(len(all_loss)) + 1), all_loss, label=f'Loss (Batch size ratio {batch_size_ratio})')
                # plt.plot((np.arange(len(all_acc)) + 1), all_acc, label='Training Accuracy')
                plt.title('Training Curve')
                plt.xlabel('Epochs')
                plt.legend()
        if plot:
            plt.show()

    def train_diff_lr(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size_ratio: float,
                      early_start_acc: Optional[Tuple[float, int]] = None, display_freq: int = 1,
                      plot: bool = False,
                      lr_list: list = None):
        for lr in lr_list:
            self._init_weights()
            if batch_size_ratio > 1. or batch_size_ratio <= 0:
                raise ValueError(f'Batch size must be <= 1 and > 0 but got {batch_size_ratio}')

            n_samples = x.shape[0]
            batch_size = int(n_samples * batch_size_ratio)
            n_batches = int(np.ceil(n_samples / batch_size))
            last_batch_size = n_samples - ((n_batches - 1) * batch_size)

            n_reach_early_stop_acc = 0
            n_accumulated_epochs = 0

            all_loss = []
            all_acc = []

            for epoch in range(1, epochs + 1):
                n_accumulated_epochs += 1
                if n_accumulated_epochs >= display_freq:
                    print(f'=== epoch {epoch} (lr={lr}) === ', end='')
                indices = np.random.choice(n_samples, n_samples, replace=False)
                for batch in range(1, n_batches + 1):
                    self._init_grad()
                    if batch == n_batches:
                        cur_batch_size = deepcopy(last_batch_size)
                    else:
                        cur_batch_size = deepcopy(batch_size)
                    start_idx = (batch - 1) * batch_size
                    end_idx = start_idx + cur_batch_size
                    batch_x, batch_y = x[indices[start_idx:end_idx]], y[indices[start_idx:end_idx]]
                    #
                    for one_x, one_y in zip(batch_x, batch_y):
                        one_pred = self._forward(one_x)
                        self._backward(y_true=one_y, y_pred=one_pred)
                    # Update weights
                    self.fc1w -= lr * self.sum_fc1d / cur_batch_size
                    self.fc2w -= lr * self.sum_fc2d / cur_batch_size
                    self.fc3w -= lr * self.sum_fc3d / cur_batch_size
                # Evaluate
                loss = 0.0
                correct = 0
                for one_x, one_y in zip(x, y):  # one_y is 1-dim; one_pred is 2-dim
                    one_pred = self._forward(one_x)
                    loss += np.power(one_y[0] - one_pred[0][0], 2)
                    pred_class = 1. if one_pred[0][0] >= 0.5 else 0.
                    if pred_class == one_y[0]:
                        correct += 1
                loss /= n_samples
                acc = correct / n_samples
                all_loss.append(loss)
                all_acc.append(acc)
                if n_accumulated_epochs >= display_freq:
                    print(f'Loss: {loss:.6f} ', end='')
                    print(f'Training Accuracy: {acc:.6f}')
                    n_accumulated_epochs = 0
                if early_start_acc:
                    if acc > early_start_acc[0]:
                        n_reach_early_stop_acc += 1
                        if n_reach_early_stop_acc > early_start_acc[1]:
                            print(f'> {early_start_acc} accuracy early stop')
                            break
                    else:
                        n_reach_early_stop_acc = 0
            if plot:
                plt.plot((np.arange(len(all_loss)) + 1), all_loss, label=f'Loss (lr={lr})')
                # plt.plot((np.arange(len(all_acc)) + 1), all_acc, label='Training Accuracy')
                plt.title('Training Curve')
                plt.xlabel('Epochs')
                plt.legend()
        if plot:
            plt.show()
