from collections import namedtuple

from transformer import *


def pad_begin(input, k):
    shp = tf.shape(input)
    return tf.concat([tf.ones([shp[0], k], dtype=tf.int32) * 2, input], 1)


def decoder_self_attention_bias(length, k):
    max_length = 500
    max_length = (max_length // k) * k
    m = np.zeros([max_length, max_length], dtype=np.float32)
    for i in xrange(max_length // k):
        m[i * k: i * k + k, :i * k + k] = 1.0
    m = tf.convert_to_tensor(m)
    m = m[:length, :length]
    ret = -1e9 * (1.0 - m)
    return tf.reshape(ret, [1, 1, length, length])


class PTransformer(Transformer):
    def __init__(self, *args, **kargs):
        super(PTransformer, self).__init__(*args, **kargs)
        self._use_cache = True

    def decoder_impl(self, decoder_input, encoder_output, is_training):
        attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

        num_parallel = self._config.num_parallel
        padded_decoder_input = pad_begin(decoder_input, num_parallel - 1)
        length = tf.floor_div(tf.shape(decoder_input)[1] + self._config.num_parallel - 1,
                              self._config.num_parallel) * self._config.num_parallel
        padded_decoder_input = padded_decoder_input[:, :length]

        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)
        decoder_output = embedding(padded_decoder_input,
                                   vocab_size=self._config.dst_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   kernel=self._dst_embedding,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="dst_embedding")
        # Positional Encoding
        decoder_output = common_attention.add_timing_signal_1d(decoder_output)

        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=residual_dropout_rate,
                                           training=is_training)
        # Bias for preventing peeping later information
        self_attention_bias = decoder_self_attention_bias(tf.shape(decoder_output)[1], self._config.num_parallel)
        # Blocks
        for i in range(self._config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=self_attention_bias,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              output_depth=self._config.hidden_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              output_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              name="decoder_vanilla_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Position-wise Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=self._config.ff_hidden_units,
                                              output_size=self._config.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=residual_dropout_rate)

        decoder_output = decoder_output[:, :tf.shape(decoder_input)[1]]

        return decoder_output

    def decoder_with_caching_impl(self, decoder_input, decoder_cache, encoder_output, is_training):

        attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)

        padded_decoder_input = pad_begin(decoder_input, self._config.num_parallel - 1)
        decoder_output = embedding(padded_decoder_input,
                                   vocab_size=self._config.dst_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   kernel=self._dst_embedding,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="dst_embedding")

        # Positional Encoding
        decoder_output = common_attention.add_timing_signal_1d(decoder_output)

        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=residual_dropout_rate,
                                           training=is_training)

        num_parallel = self._config.num_parallel
        new_cache = []

        # Blocks
        for i in range(self._config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output[:, -num_parallel:, :],
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=None,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              num_queries=num_parallel,
                                              output_depth=self._config.hidden_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              output_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              num_queries=num_parallel,
                                              name="decoder_vanilla_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Position-wise Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=self._config.ff_hidden_units,
                                              output_size=self._config.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=residual_dropout_rate)

                decoder_output = tf.concat([decoder_cache[:, :, i, :], decoder_output], axis=1)
                new_cache.append(decoder_output[:, :, None, :])

        new_cache = tf.concat(new_cache, axis=2)  # [batch_size, n_step, num_blocks, num_hidden]

        return decoder_output, new_cache

    def test_output_multiple(self, decoder_output, k, reuse):
        """Predict num_parallel tokens at once."""

        num_parallel = self._config.num_parallel
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            last_logits = dense(decoder_output[:, -num_parallel:], self._config.dst_vocab_size, use_bias=False,
                                kernel=self._dst_softmax, name='dst_softmax', reuse=None)
            next_pred = tf.to_int32(tf.argmax(last_logits, axis=-1))  # [B, P]
            z = tf.nn.log_softmax(last_logits)
            next_scores, next_preds = tf.nn.top_k(z, k=k, sorted=False)  # [B, P, K]
            next_preds = tf.to_int32(next_preds)

        return next_pred, next_preds, next_scores

    def beam_search(self, encoder_output, use_cache, reuse):
        """Beam search in graph."""
        beam_size, batch_size = self._config.test.beam_size, tf.shape(encoder_output)[0]

        if beam_size == 1:
            return self.greedy_search(encoder_output, use_cache, reuse)

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
            bias = tf.to_float(bias)
            b = tf.constant([0.0] + [-inf] * (beam_size - 1))
            b = tf.tile(b[None, :], multiples=[batch_size * beam_size, 1])
            return scores * (1 - bias[:, None]) + b * bias[:, None]

        def get_bias_preds(preds, bias):
            """
            If a sequence is finished, all of its branch should be </S> (3).
            Args:
                preds: A int array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A int array with shape [batch_size * beam_size].
            """
            bias = tf.to_int32(bias)
            return preds * (1 - bias[:, None]) + bias[:, None] * 3

        # Prepare beam search inputs.
        # [batch_size, 1, *, hidden_units]
        encoder_output = encoder_output[:, None, :, :]
        # [batch_size, beam_size, *, hidden_units]
        encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
        encoder_output = tf.reshape(encoder_output, [batch_size * beam_size, -1, encoder_output.get_shape()[-1].value])
        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = tf.ones([batch_size * beam_size, 1], dtype=tf.int32) * 2
        scores = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)  # [beam_size]
        scores = tf.tile(scores, multiples=[batch_size])  # [batch_size * beam_size]
        lengths = tf.zeros([batch_size * beam_size], dtype=tf.float32)
        bias = tf.zeros_like(scores, dtype=tf.bool)
        Cache = namedtuple('Cache', ['decoder_cache', 'next_preds', 'next_scores'])
        caches = Cache(
            decoder_cache=tf.zeros([batch_size * beam_size, 0, self._config.num_blocks, self._config.hidden_units]),
            next_preds=tf.zeros([batch_size * beam_size, 0, self._config.test.beam_size], dtype=tf.int32),
            next_scores=tf.zeros([batch_size * beam_size, 0, self._config.test.beam_size]))

        def step(i, bias, preds, scores, lengths, caches):
            # Where are we.
            i += 1

            # Call decoder and get predictions.
            if use_cache:

                def compute():
                    decoder_output, decoder_cache = \
                        self.decoder_with_caching(preds, caches.decoder_cache, encoder_output,
                                                  is_training=False, reuse=reuse)
                    _, next_preds, next_scores = self.test_output_multiple(decoder_output,
                                                                           k=self._config.test.beam_size,
                                                                           reuse=reuse)
                    new_caches = Cache(decoder_cache=decoder_cache, next_preds=next_preds, next_scores=next_scores)
                    return new_caches

                def hit():
                    return caches

                cond = tf.equal(tf.shape(caches.next_preds)[1], 0)
                caches = tf.cond(cond, compute, hit)
                next_preds = caches.next_preds[:, 0]
                next_scores = caches.next_scores[:, 0]
                caches = Cache(decoder_cache=caches.decoder_cache,
                               next_preds=caches.next_preds[:, 1:],
                               next_scores=caches.next_scores[:, 1:])
            else:
                decoder_output = self.decoder(preds, encoder_output, is_training=False, reuse=reuse)
                _, next_preds, next_scores = self.test_output(decoder_output, reuse=reuse)

            next_preds = get_bias_preds(next_preds, bias)
            next_scores = get_bias_scores(next_scores, bias)

            # Update scores.
            scores = scores[:, None] + next_scores  # [batch_size * beam_size, beam_size]
            scores = tf.reshape(scores, shape=[batch_size, beam_size ** 2])  # [batch_size, beam_size * beam_size]

            # LP scores.
            lengths = lengths[:, None] + tf.to_float(tf.not_equal(next_preds, 3))  # [batch_size * beam_size, beam_size]
            lengths = tf.reshape(lengths, shape=[batch_size, beam_size ** 2])  # [batch_size, beam_size * beam_size]
            lp = tf.pow((5 + lengths) / (5 + 1), self._config.test.lp_alpha)  # Length penalty
            lp_scores = scores / lp  # following GNMT

            # Pruning
            _, k_indices = tf.nn.top_k(lp_scores, k=beam_size)
            base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
            base_indices *= beam_size ** 2
            k_indices = base_indices + tf.reshape(k_indices, shape=[-1])  # [batch_size * beam_size]

            # Update lengths.
            lengths = tf.reshape(lengths, [-1])
            lengths = tf.gather(lengths, k_indices)

            # Update scores.
            scores = tf.reshape(scores, [-1])
            scores = tf.gather(scores, k_indices)

            # Update predictions.
            next_preds = tf.gather(tf.reshape(next_preds, shape=[-1]), indices=k_indices)
            preds = tf.gather(preds, indices=k_indices / beam_size)
            if use_cache:
                caches = Cache(decoder_cache=tf.gather(caches.decoder_cache, indices=k_indices / beam_size),
                               next_preds=tf.gather(caches.next_preds, indices=k_indices / beam_size),
                               next_scores=tf.gather(caches.next_scores, indices=k_indices / beam_size))
            preds = tf.concat((preds, next_preds[:, None]), axis=1)  # [batch_size * beam_size, i]

            # Whether sequences finished.
            bias = tf.equal(preds[:, -1], 3)  # </S>?

            return i, bias, preds, scores, lengths, caches

        def not_finished(i, bias, preds, scores, lengths, caches):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(bias)),
                tf.less_equal(
                    i,
                    tf.reduce_min([tf.shape(encoder_output)[1] + 50, self._config.test.max_target_length])
                )
            )

        i, bias, preds, scores, lengths, caches = \
            tf.while_loop(cond=not_finished,
                          body=step,
                          loop_vars=[0, bias, preds, scores, lengths, caches],
                          shape_invariants=[
                              tf.TensorShape([]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None]),
                              Cache(decoder_cache=tf.TensorShape([None, None, None, None]),
                                    next_preds=tf.TensorShape([None, None, None]),
                                    next_scores=tf.TensorShape([None, None, None]))
                              ],
                          back_prop=False)

        scores = tf.reshape(scores, shape=[batch_size, beam_size])
        preds = tf.reshape(preds, shape=[batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]

        max_indices = tf.to_int32(tf.argmax(scores, axis=-1))  # [batch_size]
        max_indices += tf.range(batch_size) * beam_size
        preds = tf.reshape(preds, shape=[batch_size * beam_size, -1])

        final_preds = tf.gather(preds, indices=max_indices)
        final_preds = final_preds[:, 1:]  # remove <S> flag
        return final_preds

    def greedy_search(self, encoder_output, use_cache, reuse):
        """Beam search in graph."""
        batch_size = tf.shape(encoder_output)[0]
        num_parallel = self._config.num_parallel

        preds = tf.ones([batch_size, 1], dtype=tf.int32) * 2
        scores = tf.zeros([batch_size], dtype=tf.float32)
        finished = tf.zeros([batch_size], dtype=tf.bool)
        cache = tf.zeros([batch_size, 0, self._config.num_blocks, self._config.hidden_units])

        def step(i, finished, preds, scores, cache):
            # Where are we.
            i += num_parallel

            # Call decoder and get predictions.
            decoder_output, cache = self.decoder_with_caching(preds, cache, encoder_output, is_training=False, reuse=reuse)
            _, next_preds, next_scores = self.test_output_multiple(decoder_output, k=1, reuse=reuse)
            next_preds = next_preds[:, :, 0]
            next_scores = tf.reduce_sum(next_scores[:, :, 0], axis=1)

            # Update.
            scores = scores + next_scores
            preds = tf.concat([preds, next_preds], axis=1)

            # Whether sequences finished.
            has_eos = tf.reduce_any(tf.equal(next_preds, 3), axis=1)
            finished = tf.logical_or(finished, has_eos)

            return i, finished, preds, scores, cache

        def not_finished(i, finished, preds, scores, cache):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(finished)),
                tf.less_equal(
                    i,
                    tf.reduce_min([tf.shape(encoder_output)[1] + 50, self._config.test.max_target_length])
                )
            )

        i, finished, preds, scores, cache = \
            tf.while_loop(cond=not_finished,
                          body=step,
                          loop_vars=[0, finished, preds, scores, cache],
                          shape_invariants=[
                              tf.TensorShape([]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None, None, None])],
                          back_prop=False)

        preds = preds[:, 1:]  # remove <S> flag
        return preds

    # def test_output_beam(self, decoder_output, reuse):
    #     beam_size = self._config.test.beam_size
    #     num_parallel = self._config.num_parallel
    #     eos = 3
    #     k = int(np.ceil(np.power(beam_size, 1.0 / num_parallel)))
    #     _, next_preds, next_scores = self.test_output_multiple(decoder_output, k, reuse)  # [batch_size, p, k]
    #
    #     batch_size = tf.shape(decoder_output)[0]
    #     scores = next_scores[:, 0, :]  # [batch_size, k**1]
    #     preds = next_preds[:, 0, :, None]  # [batch_size, k**1, 1]
    #     finished = np.zeros_like(scores)  # [batch_size, k**1]
    #
    #     def get_biased_scores(scores, finished):
    #         pass
    #
    #     def get_biased_preds(preds, finished):
    #         pass
    #
    #     for i in range(1, self._config.num_parallel):
    #         cur_preds = next_preds[:, i, :]  # [batch_size, k]
    #         cur_scores = next_scores[:, i, :]  # [batch_size, k]
    #
    #         finished = finished[:, :, None]
    #         finished = tf.tile(finished, [1, 1, k])  # [batch_size, k**i, k]
    #         finished = tf.mul(finished, tf.to_float(tf.equal(cur_preds, eos)[:, None, :]))  # [batch_size, k**i, k]
    #         # finished = tf.reshape(finished, [batch_size, -1])  # [batch_size, k**(i+1)]
    #
    #         scores = scores[:, :, None]
    #         scores = tf.tile(scores, [1, 1, k])
    #         scores += finished * cur_scores[:, None, :]
    #         scores = tf.reshape(scores, [batch_size, -1])
    #
    #         preds = preds[:, :, None, :]
    #         preds = tf.tile(preds, [1, 1, k, 1])  # [batch_size, k**i, k, i]
    #         preds = tf.reshape(preds, [batch_size, k ** (i+1), -1])  # [batch_size, k**(i+1), i]
    #         cur_preds = tf.tile(cur_preds[:, :, None], [1, k ** i, 1])  # [batch_size, k**(i+1), 1]
    #         preds = tf.concat([preds, cur_preds], axis=2)  # [batch_size, k**(i+1), i+1]
    #
    #     # Select the top beam_size predictions.
    #     top_scores, top_indices = tf.nn.top_k(scores, beam_size)
    #     flatten_top_indices = tf.reshape(top_indices, [-1])
    #     base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
    #     base_indices *= k**num_parallel
    #     flatten_top_indices += base_indices
    #     flatten_preds = tf.reshape(preds, [-1, num_parallel])
    #     top_preds = tf.gather(flatten_preds, flatten_top_indices)
    #     top_preds = tf.reshape(top_preds, [batch_size, beam_size, num_parallel])
    #
    #     return top_preds, top_scores    # [batch_size, beam_size, num_parallel], [batch_size, beam_size]

    # def faster_beam_search(self, encoder_output, use_cache, reuse):
    #     """Beam search in graph."""
    #     beam_size, batch_size = self._config.test.beam_size, tf.shape(encoder_output)[0]
    #     inf = 1e10
    #
    #     def get_bias_scores(scores, bias):
    #         """
    #         If a sequence is finished, we only allow one alive branch.
    #         This function aims to give one branch a zero score and the rest -inf score.
    #         Args:
    #             scores: A real value array with shape [batch_size * beam_size, beam_size].
    #             bias: A bool array with shape [batch_size * beam_size].
    #
    #         Returns:
    #             A real value array with shape [batch_size * beam_size, beam_size].
    #         """
    #         bias = tf.to_float(bias)
    #         b = tf.constant([0.0] + [-inf] * (beam_size - 1))
    #         b = tf.tile(b[None, :], multiples=[batch_size * beam_size, 1])
    #         return scores * (1 - bias[:, None]) + b * bias[:, None]
    #
    #     def get_bias_preds(preds, bias):
    #         """
    #         If a sequence is finished, all of its branch should be </S> (3).
    #         Args:
    #             preds: A int array with shape [batch_size * beam_size, beam_size].
    #             bias: A bool array with shape [batch_size * beam_size].
    #
    #         Returns:
    #             A int array with shape [batch_size * beam_size].
    #         """
    #         bias = tf.to_int32(bias)
    #         return preds * (1 - bias[:, None]) + bias[:, None] * 3
    #
    #     # Prepare beam search inputs.
    #     # [batch_size, 1, *, hidden_units]
    #     encoder_output = encoder_output[:, None, :, :]
    #     # [batch_size, beam_size, *, hidden_units]
    #     encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
    #     encoder_output = tf.reshape(encoder_output, [batch_size * beam_size, -1, encoder_output.get_shape()[-1].value])
    #     # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
    #     preds = tf.ones([batch_size * beam_size, 1], dtype=tf.int32) * 2
    #     scores = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)  # [beam_size]
    #     scores = tf.tile(scores, multiples=[batch_size])  # [batch_size * beam_size]
    #     lengths = tf.zeros([batch_size * beam_size], dtype=tf.float32)
    #     bias = tf.zeros_like(scores, dtype=tf.bool)
    #
    #     if use_cache:
    #         cache = tf.zeros([batch_size * beam_size, 0, self._config.num_blocks, self._config.hidden_units])
    #     else:
    #         cache = tf.zeros([0, 0, 0, 0])
    #
    #     def step(i, bias, preds, scores, lengths, cache):
    #         # Where are we.
    #         i += 1
    #
    #         # Call decoder and get predictions.
    #         if use_cache:
    #             decoder_output, cache = \
    #                 self.decoder_with_caching(preds, cache, encoder_output, is_training=False, reuse=reuse)
    #         else:
    #             decoder_output = self.decoder(preds, encoder_output, is_training=False, reuse=reuse)
    #
    #         # next_preds: [batch_size, beam_size, num_parallel]
    #         # next_scores: [batch_size, beam_size]
    #         _, next_k_preds, next_k_scores = self.test_output(decoder_output, reuse=reuse)
    #
    #         next_preds = get_bias_preds(next_preds, bias)
    #         next_scores = get_bias_scores(next_scores, bias)
    #
    #         # Update scores.
    #         scores = scores[:, None] + next_scores  # [batch_size * beam_size, beam_size]
    #         scores = tf.reshape(scores, shape=[batch_size, beam_size ** 2])  # [batch_size, beam_size * beam_size]
    #
    #         # LP scores.
    #         lengths = lengths[:, None] + tf.to_float(tf.not_equal(next_preds, 3))  # [batch_size * beam_size, beam_size]
    #         lengths = tf.reshape(lengths, shape=[batch_size, beam_size ** 2])  # [batch_size, beam_size * beam_size]
    #         lp = tf.pow((5 + lengths) / (5 + 1), self._config.test.lp_alpha)  # Length penalty
    #         lp_scores = scores / lp  # following GNMT
    #
    #         # Pruning
    #         _, k_indices = tf.nn.top_k(lp_scores, k=beam_size)
    #         base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
    #         base_indices *= beam_size ** 2
    #         k_indices = base_indices + tf.reshape(k_indices, shape=[-1])  # [batch_size * beam_size]
    #
    #         # Update lengths.
    #         lengths = tf.reshape(lengths, [-1])
    #         lengths = tf.gather(lengths, k_indices)
    #
    #         # Update scores.
    #         scores = tf.reshape(scores, [-1])
    #         scores = tf.gather(scores, k_indices)
    #
    #         # Update predictions.
    #         next_preds = tf.gather(tf.reshape(next_preds, shape=[-1]), indices=k_indices)
    #         preds = tf.gather(preds, indices=k_indices / beam_size)
    #         if use_cache:
    #             cache = tf.gather(cache, indices=k_indices / beam_size)
    #         preds = tf.concat((preds, next_preds[:, None]), axis=1)  # [batch_size * beam_size, i]
    #
    #         # Whether sequences finished.
    #         bias = tf.equal(preds[:, -1], 3)  # </S>?
    #
    #         return i, bias, preds, scores, lengths, cache
    #
    #     def not_finished(i, bias, preds, scores, lengths, cache):
    #         return tf.logical_and(
    #             tf.reduce_any(tf.logical_not(bias)),
    #             tf.less_equal(
    #                 i,
    #                 tf.reduce_min([tf.shape(encoder_output)[1] + 50, self._config.test.max_target_length])
    #             )
    #         )
    #
    #     i, bias, preds, scores, lengths, cache = \
    #         tf.while_loop(cond=not_finished,
    #                       body=step,
    #                       loop_vars=[0, bias, preds, scores, lengths, cache],
    #                       shape_invariants=[
    #                           tf.TensorShape([]),
    #                           tf.TensorShape([None]),
    #                           tf.TensorShape([None, None]),
    #                           tf.TensorShape([None]),
    #                           tf.TensorShape([None]),
    #                           tf.TensorShape([None, None, None, None])],
    #                       back_prop=False)
    #
    #     scores = tf.reshape(scores, shape=[batch_size, beam_size])
    #     preds = tf.reshape(preds, shape=[batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]
    #
    #     max_indices = tf.to_int32(tf.argmax(scores, axis=-1))  # [batch_size]
    #     max_indices += tf.range(batch_size) * beam_size
    #     preds = tf.reshape(preds, shape=[batch_size * beam_size, -1])
    #
    #     final_preds = tf.gather(preds, indices=max_indices)
    #     final_preds = final_preds[:, 1:]  # remove <S> flag
    #     return final_preds

    # def test_loss(self, decoder_output, Y, reuse):
    #     """This function help users to compute PPL during test."""
    #     with tf.variable_scope(self.decoder_scope, reuse=reuse):
    #         logits = dense(decoder_output, self._config.dst_vocab_size, use_bias=False,
    #                        kernel=self._dst_softmax, name="decoder", reuse=None)
    #         mask = tf.to_float(tf.not_equal(Y, 0))
    #         labels = tf.one_hot(Y, depth=self._config.dst_vocab_size)
    #         loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    #         loss_sum = tf.reduce_sum(loss * mask)
    #         # Position-wise PPL
    #         lengths = tf.reduce_sum(mask, axis=1, keep_dims=True)
    #         lengths_mask = tf.to_float(tf.greater(lengths, 12))
    #         loss_sum = tf.Print(loss_sum,
    #                             [tf.reduce_sum(loss * lengths_mask, axis=0)[:12] /
    #                              (tf.reduce_sum(lengths_mask) + 1e-6)],
    #                             summarize=15)
    #         probs = tf.nn.softmax(logits)
    #     return loss_sum, probs

    def train_output(self, decoder_output, Y, teacher_probs, reuse):
        """Calculate loss and accuracy."""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            logits = dense(decoder_output, self._config.dst_vocab_size, use_bias=False,
                           kernel=self._dst_softmax, name='decoder', reuse=None)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            mask = tf.to_float(tf.not_equal(Y, 0))

            # Token-level accuracy
            acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * mask) / tf.reduce_sum(mask)
            if not tf.get_variable_scope().reuse:
                tf.summary.scalar('accuracy', acc)

            if teacher_probs is not None:
                # Knowledge distillation
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=teacher_probs)
            else:
                # Smoothed loss
                loss = common_layers.smoothing_cross_entropy(logits=logits, labels=Y,
                                                             vocab_size=self._config.dst_vocab_size,
                                                             confidence=1 - self._config.train.label_smoothing)
            loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

            self.register_loss('ml_loss', loss)
