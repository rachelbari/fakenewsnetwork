import sys
import os
import numpy as np
import math
import time
import itertools
import shutil
import tensorflow as tf
import tree as tr
from utils import Vocab

RESET_AFTER = 50


class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    embed_size = 100
    label_size = 2
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 40
    lr = 0.01
    l2 = 0.02
    model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights' % (embed_size, l2, lr)


class RNN_Model():
    def load_data(self):
        """Loads train/dev/test data and builds vocabulary."""
        self.train_data, self.dev_data, self.test_data = tr.simplified_data(
            300, 70, 100)

        # build vocab from training data
        self.vocab = Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

        self.w2v_vocab, w2v_embd, embedding_dict = self.load_w2v()
        self.embedding_dim = len(w2v_embd[0])
        self.w2v_vocab_size = len(self.w2v_vocab)

        self.vocab_size = len(self.vocab)
        embeddings_tmp = []
        for i in range(self.vocab_size):
            item = self.vocab.decode(i)
            if item in self.w2v_vocab:
                embeddings_tmp.append(embedding_dict[item])
                # print("Found word {}".format(item))
            else:
                # print("Couldn't find {}.".format(item))
                rand_num = np.random.uniform(
                    low=-0.2, high=0.2, size=self.embedding_dim)
                embeddings_tmp.append(rand_num)

        self.embed = np.asarray(embeddings_tmp)

    def inference(self, tree, predict_only_root=True):
        """For a given tree build the RNN models computation graph up to where it
            may be used for inference.
        Args:
            tree: a Tree object on which to build the computation graph for the RNN
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        node_tensors = self.add_model(tree.root)
        if predict_only_root:
            node_tensors = node_tensors[tree.root]
        else:
            node_tensors = [
                tensor for node, tensor in node_tensors.items()
                if node.label != 2
            ]
            node_tensors = tf.concat(node_tensors, 0)
        return self.add_projections(node_tensors)

    def add_model_vars(self):
        '''
        You model contains the following parameters:
            embedding:  tensor(vocab_size, embed_size)
            W1:         tensor(2* embed_size, embed_size)
            b1:         tensor(1, embed_size)
            U:          tensor(embed_size, output_size)
            bs:         tensor(1, output_size)
        Hint: Add the tensorflow variables to the graph here and *reuse* them while building
                the compution graphs for composition and projection for each tree
        Hint: Use a variable_scope "Composition" for the composition layer, and
              "Projection") for the linear transformations preceding the softmax.
        Hint: Look up tf.get_variable
        '''
        with tf.variable_scope('Composition'):
            ### YOUR CODE HERE
            # embedding = tf.get_variable(
            #     "embedding", (len(self.vocab), self.config.embed_size))

            embedding = tf.get_variable(
                "embedding",
                shape=[self.vocab_size, self.embedding_dim],
                initializer=tf.constant_initializer(self.embed),
                trainable=False)
            # embedding = tf.Variable(
            #     tf.constant(0.0, shape=[self.vocab_size, self.embedding_dim]),
            #     trainable=False,
            #     name="embedding")
            # self.embedding_placeholder = tf.placeholder(
            #     tf.float32, [self.vocab_size, self.embedding_dim])
            # self.embedding_init = embedding.assign(self.embedding_placeholder)

            # embedding = tf.get_variable("embedding", shape=[self.w2v_vocab_size, self.config.embed_size],
            #             initializaer=tf.constant_initializer(self.embed), trainable=False)
            W1 = tf.get_variable("W1",
                                 (self.embedding_dim, self.embedding_dim))
            b1 = tf.get_variable("b1", (1, self.embedding_dim))
            ### END YOUR CODE
        with tf.variable_scope('Projection'):
            ### YOUR CODE HERE
            U = tf.get_variable("U",
                                (self.embedding_dim, self.config.label_size))
            bs = tf.get_variable("bs", (1, self.config.label_size))
            ### END YOUR CODE

    def add_model(self, node):
        """Recursively build the model to compute the phrase embeddings in the tree

        Hint: Refer to tree.py and vocab.py before you start. Refer to
              the model's vocab with self.vocab
        Hint: Reuse the "Composition" variable_scope here
        Hint: Store a node's vector representation in node.tensor so it can be
              used by its parent
        Hint: If node is a leaf node, it's vector representation is just that of the
              word vector (see tf.gather()).
        Args:
            node: a Node object
        Returns:
            node_tensors: Dict: key = Node, value = tensor(1, embed_size)
        """
        with tf.variable_scope('Composition', reuse=True):
            ### YOUR CODE HERE
            embedding = tf.get_variable("embedding")
            W1 = tf.get_variable("W1")
            b1 = tf.get_variable("b1")
            # the variables are already stored in self?
        ## END YOUR CODE

        node_tensors = dict()
        curr_node_tensor = None
        if node.isLeaf:
            ### YOUR CODE HERE
            # word_id = self.vocab.encode(node.word)
            # embedded_chars = tf.nn.embedding_lookup(embedding, word_id)
            # curr_node_tensor = tf.unstack(embedded_chars, 1, 1)

            word_id = self.vocab.encode(node.word)
            curr_node_tensor = tf.expand_dims(tf.gather(embedding, word_id), 0)
            ### END YOUR CODE
        else:
            node_input = tf.zeros((1, self.embedding_dim))
            for child in node.children:
                node_tensors.update(self.add_model(child))
                node_input = tf.add(node_input, node_tensors[child])
            ### YOUR CODE HERE
            curr_node_tensor = tf.nn.relu(tf.matmul(node_input, W1) + b1)
            ### END YOUR CODE
        node_tensors[node] = curr_node_tensor
        return node_tensors

    def add_projections(self, node_tensors):
        """Add projections to the composition vectors to compute the raw sentiment scores

        Hint: Reuse the "Projection" variable_scope here
        Args:
            node_tensors: tensor(?, embed_size)
        Returns:
            output: tensor(?, label_size)
        """
        logits = None
        ### YOUR CODE HERE
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable("U")
            bs = tf.get_variable("bs")
            logits = tf.matmul(node_tensors, U) + bs
        ### END YOUR CODE
        return logits

    def loss(self, logits, labels):
        """Adds loss ops to the computational graph.

        Hint: Use sparse_softmax_cross_entropy_with_logits
        Hint: Remember to add l2_loss (see tf.nn.l2_loss)
        Args:
            logits: tensor(num_nodes, output_size)
            labels: python list, len = num_nodes
        Returns:
            loss: tensor 0-D
        """
        loss = None
        # YOUR CODE HERE
        with tf.variable_scope('Composition', reuse=True):
            W1 = tf.get_variable("W1")

        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable("U")

        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits)) + self.config.l2 * tf.nn.l2_loss(
                    W1) + self.config.l2 * tf.nn.l2_loss(U)

        # END YOUR CODE
        return loss

    def training(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.GradientDescentOptimizer for this model.
                Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: tensor 0-D
        Returns:
            train_op: tensorflow op for training.
        """
        train_op = None
        # YOUR CODE HERE
        train_op = tf.train.GradientDescentOptimizer(
            self.config.lr).minimize(loss)
        # END YOUR CODE
        return train_op

    def predictions(self, y):
        """Returns predictions from sparse scores

        Args:
            y: tensor(?, label_size)
        Returns:
            predictions: tensor(?,1)
        """
        predictions = None
        # YOUR CODE HERE
        predictions = tf.argmax(y, axis=1)
        # END YOUR CODE
        return predictions

    def __init__(self, config):
        self.config = config
        self.load_data()

    def predict(self, trees, weights_path, get_loss=False):
        """Make predictions from the provided model."""
        results = []
        losses = []
        for i in range(int(math.ceil(len(trees) / float(RESET_AFTER)))):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                # sess.run(
                #     self.embedding_init,
                #     feed_dict={self.embedding_placeholder: self.embed})
                saver = tf.train.Saver()
                saver.restore(sess, weights_path)
                for tree in trees[i * RESET_AFTER:(i + 1) * RESET_AFTER]:
                    logits = self.inference(tree, True)
                    predictions = self.predictions(logits)
                    root_prediction = sess.run(predictions)[0]
                    if root_prediction == 1:
                        root_prediction = 4
                    if get_loss:
                        root_label = tree.root.label
                        loss = sess.run(self.loss(logits, [root_label]))
                        losses.append(loss)
                    results.append(root_prediction)
        return results, losses

    def run_epoch(self, new_model=False, verbose=True):
        step = 0
        loss_history = []
        while step < len(self.train_data):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                # sess.run(
                #     self.embedding_init,
                #     feed_dict={self.embedding_placeholder: self.embed})
                if new_model:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                else:
                    saver = tf.train.Saver()
                    saver.restore(sess,
                                  './weights/%s.temp' % self.config.model_name)
                for _ in range(RESET_AFTER):
                    if step >= len(self.train_data):
                        break
                    tree = self.train_data[step]
                    logits = self.inference(tree)
                    # print(sess.run(logits))
                    labels = [l for l in tree.labels if l != 2]
                    if labels[0] == 4:
                        labels = [1]
                    # print(labels)
                    loss = self.loss(logits, labels)
                    train_op = self.training(loss)
                    loss, _ = sess.run([loss, train_op])
                    loss_history.append(loss)
                    if verbose:
                        sys.stdout.write('\r{} / {} :    loss = {}'.format(
                            step, len(self.train_data), np.mean(loss_history)))
                        sys.stdout.flush()
                    step += 1
                saver = tf.train.Saver()
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                saver.save(sess, './weights/%s.temp' % self.config.model_name)
        train_preds, _ = self.predict(
            self.train_data, './weights/%s.temp' % self.config.model_name)
        val_preds, val_losses = self.predict(
            self.dev_data,
            './weights/%s.temp1' % self.config.model_name,
            get_loss=True)
        train_labels = [t.root.label for t in self.train_data]
        val_labels = [t.root.label for t in self.dev_data]
        train_acc = np.equal(train_preds, train_labels).mean()
        val_acc = np.equal(val_preds, val_labels).mean()

        print()
        print('Training acc (only root node): {}'.format(train_acc))
        print('Validation acc (only root node): {}'.format(val_acc))
        print('Confusion matrix:')
        print(self.make_conf(train_labels, train_preds))
        print(self.make_conf(val_labels, val_preds))
        return train_acc, val_acc, loss_history, np.mean(val_losses)

    def train(self, verbose=True):
        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = -1
        for epoch in range(self.config.max_epochs):
            print('epoch %d' % epoch)
            if epoch == 0:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(
                    new_model=True)
            else:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch()
            complete_loss_history.extend(loss_history)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            #lr annealing
            epoch_loss = np.mean(loss_history)
            if epoch_loss > prev_epoch_loss * self.config.anneal_threshold:
                self.config.lr /= self.config.anneal_by
                print('annealed lr to %f' % self.config.lr)
            prev_epoch_loss = epoch_loss

            # save if model has improved on val
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch

            # if model has not improved for a while stop
            if epoch - best_val_epoch > self.config.early_stopping:
                stopped = epoch
                # break
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()

        print('\n\nstopped at %d\n' % stopped)
        return {
            'loss_history': complete_loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2, 2])
        labels = [l if l != 4 else 1 for l in labels]
        predictions = [p if p != 4 else 1 for p in predictions]
        for l, p in zip(labels, predictions):
            confmat[l, p] += 1
        return confmat

    def load_w2v(self):
        vocab = []
        embd = []
        e_dict = {}
        # change 100d to 50d for smaller-dimension GloVe embedding
        file = open("./glove.6B.100d.txt", 'r', encoding='UTF-8')
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
            e_dict[row[0]] = [float(i) for i in row[1:]]
        print("Loaded word2vec!")
        file.close()
        return vocab, embd, e_dict


def test_RNN():
    """Test RNN model implementation.

    You can use this function to test your implementation of the Sentiment
    Analysis network. When debugging, set max_epochs in the Config object to 1
    so you can rapidly iterate.
    """
    config = Config()
    model = RNN_Model(config)
    start_time = time.time()
    stats = model.train(verbose=True)
    print('Training time: {}'.format(time.time() - start_time))

    print('Test')
    print('=-=-=')
    predictions, _ = model.predict(
        model.test_data, './weights/%s.temp' % model.config.model_name)
    labels = [t.root.label for t in model.test_data]
    test_acc = np.equal(predictions, labels).mean()
    print('Test acc: {}'.format(test_acc))


if __name__ == "__main__":
    test_RNN()
