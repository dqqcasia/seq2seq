from __future__ import print_function
import codecs
import os
import tensorflow as tf
import numpy as np
import yaml
import time
import logging
from tempfile import mkstemp
from argparse import ArgumentParser

from model import Model, INT_TYPE
from utils import DataUtil, AttrDict


class Evaluator(object):
    """
    Evaluate the model.
    """
    def __init__(self, config):
        self.config = config

        # Load model
        self.model = Model(config)
        self.model.build_test_model()

        # Create session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config, graph=self.model.graph)
        # Restore model.
        with self.model.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, tf.train.latest_checkpoint(config.train.logdir))

    def __del__(self):
        self.sess.close()

    def greedy_search(self, X):
        """
        Greedy search.
        Args:
            X: A 2-d array with size [n, src_length], source sentence indices.

        Returns:
            A 2-d array with size [n, dst_length], destination sentence indices.
        """
        encoder_output = self.sess.run(self.model.encoder_output, feed_dict={self.model.src_pl: X})
        preds = np.ones([X.shape[0], 1], dtype=INT_TYPE) * 2 # <S>
        finish = np.zeros(X.shape[0:1], dtype=np.bool)
        for i in xrange(config.test.max_target_length):
            last_preds = self.sess.run(self.model.preds, feed_dict={self.model.encoder_output: encoder_output,
                                                                    self.model.decoder_input: preds})
            finish += last_preds == 3   # </S>
            if finish.all():
                break
            preds = np.concatenate((preds, last_preds[:, None]), axis=1)

        return preds[:, 1:]

    def beam_search(self, X):
        """
        Beam search with batch inputs.
        Args:
            X: A 2-d array with size [n, src_length], source sentence indices.

        Returns:
            A 2-d array with size [n, dst_length], target sentence indices.
        """

        beam_size, batch_size = config.test.beam_size, X.shape[0]
        inf = 1e10

        def get_bias_scores(scores, bias):
            """
            If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
            and the rest -inf score.
            Args:
                scores: A real value array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A real value array with shape [batch_size * beam_size, beam_size].
            """
            b = np.array([0.0] + [-inf] * (beam_size - 1))
            b = np.repeat(b[None,:], batch_size * beam_size, axis=0)  # [batch * beam_size, beam_size]
            b *= bias[:, None]
            return scores + b

        def get_bias_preds(preds, bias):
            """
            If a sequence is finished, all of its branch should be </S> (3).
            Args:
                preds: A int array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A int array with shape [batch_size * beam_size].
            """
            return preds * (1 - bias[:, None]) + bias[:, None] * 3

        # Get encoder outputs.
        encoder_output = self.sess.run(self.model.encoder_output, feed_dict={self.model.src_pl: X})
        # Prepare beam search inputs.
        encoder_output = np.repeat(encoder_output, beam_size, axis=0)   # shape: [batch_size * beam_size, hidden_units]
        preds = np.ones([batch_size * beam_size, 1], dtype=INT_TYPE) * 2  # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        scores = np.array(([0.0] + [-inf] * (beam_size - 1)) * batch_size)  # [0, -inf, -inf ,..., 0, -inf, -inf, ...], shape: [batch_size * beam_size]
        for i in xrange(config.test.max_target_length):
            # Whether sequences finished.
            bias = np.equal(preds[:, -1], 3)   # </S>?
            # If all sequences finished, break the loop.
            if bias.all():
                break

            # Expand the nodes.
            last_k_preds, last_k_scores = \
                self.sess.run([self.model.k_preds, self.model.k_scores],
                              feed_dict={self.model.encoder_output: encoder_output,
                                         self.model.decoder_input: preds})  # [batch_size * beam_size, beam_size]
            # Shrink the search range.
            scores = scores[:, None] + last_k_scores  # [batch_size * beam_size, beam_size]
            # import pdb; pdb.set_trace()
            # Bias finished sequences.
            scores = get_bias_scores(scores, bias)
            last_k_preds = get_bias_preds(last_k_preds, bias)

            scores = scores.reshape([batch_size, beam_size**2])  # [batch_size, beam_size * beam_size]

            # Reserve beam_size nodes.
            # k_indices = np.argsort(scores)[:, -beam_size:]  # [batch_size, beam_size]
            k_indices = np.argpartition(scores, kth=-beam_size, axis=-1)[:, -beam_size:]
            k_indices = np.repeat(np.array(range(0, batch_size)), beam_size) * beam_size**2 + k_indices.flatten()    # [batch_size * beam_size]
            scores = scores.flatten()[k_indices]  # [batch_size * beam_size]
            last_k_preds = last_k_preds.flatten()[k_indices]
            preds = preds[k_indices // beam_size]
            preds = np.concatenate((preds, last_k_preds[:, None]), axis=1)  # [batch_size * beam_size, i]

        scores = scores.reshape([batch_size, beam_size])
        preds = preds.reshape([batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]
        # lengths = np.sum(np.not_equal(preds, 3), axis=-1)   # [batch_size, beam_size]
        # scores /= lengths  # The simplest version of length penalty.
        max_indices = np.argmax(scores, axis=-1)   # [batch_size]
        max_indices += np.array(range(batch_size)) * beam_size
        preds = preds.reshape([batch_size * beam_size, -1])
        return preds[max_indices][:, 1:]

    def evaluate(self):
        # Load data
        du = DataUtil(self.config)
        _, tmp = mkstemp()
        fd = codecs.open(tmp, 'w', 'utf8')
        count = 0
        start = time.time()
        for batch in du.get_test_batches():
            # if config.test.beam_size == 1:
            #     Y = self.greedy_search(batch)
            # else:
            #     Y = self.beam_search(batch)
            Y = self.beam_search(batch)
            sents = du.indices_to_words(Y)
            for sent in sents:
                print(sent, file=fd)
            count += len(batch)
            logging.info('%d sentences processed in %.2f minutes.' % (count, (time.time()-start) / 60))
        fd.close()

        # Remove BPE flag, if have.
        os.system("sed -r 's/(@@ )|(@@ ?$)//g' %s > %s" % (tmp, self.config.test.output_path))

        # Call a script to evaluate.
        os.system("perl multi-bleu.perl %s < %s" % (self.config.test.dst_path, self.config.test.output_path))

                                          
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    logging.basicConfig(level=logging.DEBUG)
    evaluator = Evaluator(config)
    evaluator.evaluate()
    logging.info("Done")
