""" A neural chatbot using sequence to sequence model with
attentional decoder. 
This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
Sequence to sequence model by Cho et al.(2014)
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
This file contains the code to run the model.
See README.md for instruction on how to run the starter code.
"""

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import sys
import time
import numpy as np
import tensorflow as tf

print(tf.__version__)
from model import ChatBotModel
import config
import data
import pickle
def _get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])

def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                        " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_masks), decoder_size))

def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chat mode. """
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

def _get_buckets():
    """ Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us 
    choose a random bucket later on.
    """
    test_buckets = data.load_data('test_ids.enc', 'test_ids.dec')
    data_buckets = data.load_data('train_ids.enc', 'train_ids.dec')
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale

def _get_skip_step(iteration):
    """ How many steps should the model train before it saves all the weights. """
    if iteration < 100:
        return 50
    return 500

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")

def _eval_test_set(sess, model, test_buckets):
    """ Evaluate on the test set. """
    for bucket_id in range(len(config.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(test_buckets[bucket_id], 
                                                                        bucket_id,
                                                                        batch_size=config.BATCH_SIZE)
        _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, 
                                   decoder_masks, bucket_id, True)
        print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))

def train():
    """ Train the bot """
    test_buckets, data_buckets, train_buckets_scale = _get_buckets()
    # in train mode, we need to create the backward path, so forwrad_only is False
    model = ChatBotModel(False, config.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Running session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0
        while True:
            skip_step = _get_skip_step(iteration)
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(data_buckets[bucket_id], 
                                                                           bucket_id,
                                                                           batch_size=config.BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1
            #if iteration % 500  == 0:
                #print('Iter {}: loss {}, time {}'.format(iteration, total_loss/skip_step, time.time() - start))
                #saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.global_step)

            if iteration % skip_step == 0:
                print('Iter {}: loss {}, time {}'.format(iteration, total_loss/skip_step, time.time() - start))
                start = time.time()
                total_loss = 0
                saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.global_step)
                if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    _eval_test_set(sess, model, test_buckets)
                    start = time.time()
                sys.stdout.flush()

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])

def _construct_response(output_logits, inv_dec_vocab):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    
    #print(output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    #print(outputs)
    #print(inv_dec_vocab[outputs[0]])
    # If there is an EOS symbol in outputs, cut them at that point.
    if config.EOS_ID in outputs:
        outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    #print(" ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs]))
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])

feedbacks = [] #it will store store the user user question and provided feedback as a tuple.

def chat():
    """ in test mode, we don't to create the backward path
    """
    _, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, dec_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))
    #loading chatbot model : herre True means it will be a forward_only pass , it will not backpropogate.
    model = ChatBotModel(True, batch_size=1)
    model.build_graph()
    #initializing model checkpoints.
    saver = tf.train.Saver()

    tokens = [] #tokens will save the all the past tokens generated from user input for making chatbot remember information from the previous conversation
    ques=[] #this list will store the questionns asked by user, it will be used for feedback.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        output_file = open(os.path.join(config.PROCESSED_PATH, config.OUTPUT_FILE), 'a+')
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]
        print('Welcome to TensorBro. Say something. Enter to exit and type "feedback:" followed by correction, Max length is', max_length)

        while True:#
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n': #checking if input is there or not
                line = line[:-1]
            if line == '':#if input is null means if user has pressed 'enter' key without typing text it will break the loop.
                break
            output_file.write('HUMAN ++++ ' + line + '\n')#writing chats into text file
            
            ques.append(line)#appending text to list
            if 'feedback' in line:
            	#this is the loop which will activate if the user wants to give feedback. if user does not like the response user can write feedback: correction.
            	#for example: user>> where do you live.
            	#bot>> <unk>.
            	#user>> feedback:I live in New-York.
            	#press enter and continue chatting , you can provide multiple feedback in single session.
            	#after you end the session the bot will train the model for the feedbacks you provided.It will take time, be patient while it does that.

                feedbacks.append((ques[-2],ques[-1].split(':')[1]))#splitting user by ":" to get feedback text and passing
                print('okay i will learn the what you just said in the endo, mean while lets continue our conversation')
                line = _get_user_input()#continuing the conversation
                
            # Get token-ids for the input sentence.
            token_ids = data.sentence2id(enc_vocab, str(line))#getting the tokens for user input.for passing it to encoder.
            if len(tokens) >= max_length:
              #if the length of tokens will be greater than the bucket size it will reset the past tokens stored for converation.
              tokens = []#initializing list again
              tokens.extend(token_ids)#inserting current input
            else:
            	#if there is still space in the bucket it will extend this tokens list to feed to the encoder.
                tokens.extend(token_ids)

            if (len(token_ids) > max_length):
                print('Max length I can handle is:', max_length)
                token_ids = data.sentence2id(enc_vocab, str(line))
                line = _get_user_input()
                tokens = []
                tokens.extend(token_ids)
                continue
            # Which bucket does it belong to?
            bucket_id = _find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            #passing tokens of the current input along with the tokens of previous inputs to the model.
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(tokens, [])], 
                                                                            bucket_id,
                                                                            batch_size=1)
            
            # Get output logits for the sentence.
            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, inv_dec_vocab)
            print(response)
            output_file.write('BOT ++++ ' + response + '\n')
        output_file.write('=============================================\n')
        output_file.close()
        


def get_tokens(mode,vocab,line):
                if mode == 'dec': # we only care about '<s>' and </s> in encoder
                    ids = [vocab['<s>']]
                else:
                    ids = []
                ids.extend(data.sentence2id(vocab, line))
                # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
                if mode == 'dec':
                    ids.append(vocab['<\s>'])
                return ids

def feedback(ques):
	#feedback loop to train the corrected user inputs.
    _, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, dec_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))
    #initializing the model by passing forwad_only as false. and batch size will be equal to number user feedbacks recieved in single session
    model = ChatBotModel(False, batch_size=len(ques))
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _check_restore_parameters(sess, saver)
            # Decode from standard input.
            max_length = config.BUCKETS[-1][0]

            
            token_ids_enc = []
            #token_ids_dec = []
            for i in range(len(ques)):
            	#making user input appropriate for model input. 
              token_ids_enc.append([get_tokens('enc',enc_vocab, str(ques[i])),get_tokens('dec',dec_vocab, str(ans[i]))])
            
            
            if (len(token_ids_enc) > max_length):
                print('Max length I can handle is:', max_length)
                return
          
            bucket_id = _find_right_bucket(len(token_ids_enc))
            # Get a nummber of feedback-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(token_ids_enc, 
                                                                            bucket_id,
                                                                            batch_size=len(ques))
       
            # trainig the model for the feedback.
            _, _, _ = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, False)

          	#saving the updated model's weight.
            saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.global_step)

            print('information grasped')
            return
        
def train_joey(ques,ans):
	#feedback loop to train the corrected character.
    _, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, dec_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))
    #initializing the model by passing forwad_only as false. and batch size will be equal to number user feedbacks recieved in single session
    model = ChatBotModel(False, batch_size=config.BATCH_SIZE)
    model.build_graph()
    _, _, train_buckets_scale = _get_buckets()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]

        token_ids_enc = []
        #token_ids_dec = []
        for i in range(len(ques)):
            #making user input appropriate for model input. 
              token_ids_enc.append([get_tokens('enc',enc_vocab, str(ques[i])),get_tokens('dec',dec_vocab, str(ans[i]))])
        """ Train the bot """

        iteration = model.global_step.eval()
        total_loss = 0
        while True:
            skip_step = _get_skip_step(iteration)
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(token_ids_enc, 
                                                                           bucket_id,
                                                                           batch_size=config.BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1
            
            if iteration % skip_step == 0:
                print('Iter {}: loss {}, time {}'.format(iteration, total_loss/skip_step, time.time() - start))
                start = time.time()
                total_loss = 0
                saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.global_step)
                if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    for bucket_id in range(len(config.BUCKETS)):
                        start = time.time()
                        encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(token_ids_enc, 
                                                                                        bucket_id,
                                                                                        batch_size=config.BATCH_SIZE)
                        _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, 
                                                  decoder_masks, bucket_id, True)
                        print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))

                    start = time.time()
                sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    #selecting mode 
    parser.add_argument('--mode', choices={'train', 'chat','train_joey'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    if not os.path.isdir(config.PROCESSED_PATH):
        data.prepare_raw_data()
        data.process_data()
    print('Data ready!')
    # create checkpoints folder if there isn't one already
    data.make_dir(config.CPT_PATH)

    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
    	#chat mode activated 
        chat()
        if len(feedbacks)!=0:
        	#if there is feedback in the chat then this loop will activate
            import pickle #pickle is used to store list of input as a pickle file to further use in data.py function.
            dbfile = open('feedback','ab')
            pickle.dump(feedbacks,dbfile )
            dbfile.close() 

            tf.reset_default_graph() 
            data.prepare_raw_data(feedback=True)#it will read the data from the pickle file and add the vocabulary to the current vocab files 
            data.process_data()#it will preprocess the input.
            print('learning feedbacks.')
            feedback(feedbacks)#train the feedback after adding new words if any in the feedbacks.
            print('learned')

    elif args.mode == 'train_joey':
        questions1, answers1 = data.joey_data()
        for i in range(len(questions1)):
          questions1[i] = data.normalizeString(questions1[i])
          answers1[i] = data.normalizeString(answers1[i])

        train_joey(questions1, answers1)

if __name__ == '__main__':
    main()