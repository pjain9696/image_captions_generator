import time, math, os
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
tf.random.set_seed(1)
import matplotlib.pyplot as plt
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.filterwarnings('ignore')

from utils.load_data_utils import get_tokenizer_from_dir, get_image_to_caption_map
from utils.img_features_utils import load_image, load_image_features_on_the_fly, map_func

class Trainer:
    def __init__(self, config, vocab_size, max_len, embedding_matrix):
        self.config = config['nn_params']
        self.store_dir = config['store']
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = self.config['embedding_dim']
        self.units = self.config['units']
        self.embedding_matrix = embedding_matrix
        self.tokenizer = get_tokenizer_from_dir(config)
        self.encoder = CNN_Encoder(self.embedding_dim)
        self.decoder = RNN_Decoder(self.embedding_dim, self.units, vocab_size, embedding_matrix)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, 
            self.store_dir + self.config['checkpoint_dir'],
            max_to_keep=1
        )

    def loss_function(self, real, pred):        
        '''
        implements the SparseCategoricalCrossEntropy loss
        real: shape = (batch_size, 1) -> token values of actual word, shape = (batch_size, 1)
        pred: shape = (batch_size, vocab_size) -> prediction scores for all words in vocabulary
        '''
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype = loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        '''
        img_tensor: shape = (bathch_size, attention_features_shape, features_shape) -> encoded feature representation of images
            -> for InceptionV3, attention_features_shape = 64, features_shape = 2048 
        target: shape = (batch_size, max_len) -> actual caption tokens per batch, obtained after padding
        '''
        total_loss = 0
        #initializing the hidden state for each batch, because the captions are not related image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        #shape of hidden -> (batch_size, units)

        dec_input = tf.expand_dims([self.tokenizer.word_index['startseq']] * target.shape[0], 1)
        #shape of dec_input -> (batch_size, 1)

        with tf.GradientTape() as tape:                    
            features = self.encoder(img_tensor)
            #shape of features after coming out of encoder fc -> (batch_size, attention_features_shape, embedding_dim)

            for i in range(1, target.shape[1]):
                #passing the features through decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                #predictions shape -> (batch_size, vocab_size)

                total_loss += self.loss_function(target[:,i], predictions)

                #using teacher forcing
                dec_input = tf.expand_dims(target[:,i], 1)
        
        avg_loss = (total_loss / int(target.shape[1]))

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return avg_loss

    def validation_step(self, img_tensor, target):
        '''
        similar to method train_step but without gradient taping
        '''
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([self.tokenizer.word_index['startseq']] * target.shape[0], 1)        
        features = self.encoder(img_tensor)

        total_loss = 0
        for i in range(1, target.shape[1]):
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)
            total_loss += self.loss_function(target[:,i], predictions)

            #using teacher forcing
            dec_input = tf.expand_dims(target[:,i], 1)
        
        avg_loss = (total_loss / int(target.shape[1]))
        return avg_loss

    def initiate_training(
        self, 
        train_dataset, 
        val_dataset, 
        load_from_checkpoint=True, 
        save_loss_to_dir=True,
        load_loss_file=True  
        ):
        ''''
        main entry point to the training operation

        - train_dataset/val_dataset: a tensorflow dataset which looks like: (batch_size, (img_tensor, target_captions)):
            img_tensor: shape = (bathch_size, attention_features_shape, features_shape) -> encoded feature representation of images
            -> for InceptionV3, attention_features_shape = 64, features_shape = 2048 
            target_captions: shape = (batch_size, max_len) -> actual caption tokens per batch, obtained after padding

        - load_from_checkpoint: whether to load latest saved checkpoint in order to resume training
        - save_loss_to_dir: whether to save losses per epoch in directory (used to plot train vs val loss)
        - load_loss_file: if True, tries to read the already saved loss file and appends new losses per epoch to it 
        '''
        sample_img_tensor_val = load_image_features_on_the_fly('data/110595925_f3395c8bd6.jpg')
        caption_per_epoch_rows = []

        if load_from_checkpoint and self.checkpoint_manager.latest_checkpoint:
            print('latest_checkpoint = ', self.checkpoint_manager.latest_checkpoint)    
            start_epoch = int(self.checkpoint_manager.latest_checkpoint.split('-')[-1])
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        else:
            start_epoch = 1
        print('start_epoch =', start_epoch)

        EPOCHS = self.config['EPOCHS']
        train_loss_plot, val_loss_plot = [], []
        min_train_loss = math.inf
        for epoch in range(start_epoch, EPOCHS+1):
            start = time.time()

            #run across training set
            train_loss = 0
            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                avg_loss = self.train_step(img_tensor, target)
                train_loss += avg_loss
                if batch % 100 == 0:
                    print(f'{str(datetime.now())} Epoch {epoch} Train Batch {batch} Loss {avg_loss:.4f}')
            num_steps = train_dataset.cardinality().numpy()
            avg_train_loss_this_epoch = train_loss / num_steps

            #check val_loss
            print(f'{str(datetime.now())} computing loss on validation set...')
            val_loss = 0
            for (batch, (img_tensor, target)) in enumerate(val_dataset):
                avg_loss = self.validation_step(img_tensor, target)
                val_loss += avg_loss
            num_steps = val_dataset.cardinality().numpy()
            avg_val_loss_this_epoch = val_loss / num_steps

            #storing the epoch end loss values to plot later
            train_loss_plot.append(avg_train_loss_this_epoch.numpy())
            val_loss_plot.append(avg_val_loss_this_epoch.numpy())

            #if this epoch has lowest loss till now, save a checkpoint
            if avg_train_loss_this_epoch < min_train_loss:
                print(f'this epoch has least loss, saving a checkpoint, epoch = {epoch}, loss = {avg_train_loss_this_epoch}')
                self.checkpoint_manager.save()
                min_train_loss = avg_train_loss_this_epoch
            
            #check the caption on sample image
            this_epoch_pred_cap, _ = self.greedy_search_pred(sample_img_tensor_val, compute_attention_plot=False)
            caption_per_epoch_rows.append([epoch, this_epoch_pred_cap])
            print(f'Epoch {epoch} Average Train Loss {avg_train_loss_this_epoch:.6f}, Average Val Loss {avg_val_loss_this_epoch:.6f}, Time taken {time.time()-start:.2f}sec\n')

        if save_loss_to_dir:
            loss_df = pd.DataFrame({
                'epoch':list(range(start_epoch, EPOCHS+1)), 
                'train_loss':train_loss_plot, 
                'val_loss':val_loss_plot
            })
            loss_dir = self.store_dir + self.config['loss_dir']
            if load_loss_file and os.path.exists(loss_dir):
                saved_loss_df = pd.read_csv(loss_dir)
                saved_epochs = set(saved_loss_df['epoch'])
                #remove saved epochs from loss_df:
                loss_df = loss_df[~loss_df['epoch'].isin(saved_epochs)]
                loss_df = pd.concat([saved_loss_df, loss_df]).reset_index(drop=True)
            #save the loss_df to a file
            loss_df.to_csv(loss_dir, index=False)

        #save predicted caption per epoch to dir
        caption_per_epoch_df = pd.DataFrame(caption_per_epoch_rows, columns=['epoch', 'pred_cap'])
        caption_per_epoch_df.to_csv(self.store_dir + self.config['pred_df_dir'] + 'pred_cap_per_epoch.csv', index=False)

        #plot the losses
        # plt.plot(loss_plot)
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Loss Plot')
        # plt.grid()
        # plt.show()
        return

    def beam_search_pred(self, img_tensor_val, beam_index=3):
        '''
        predict captions from an already trained encoder-decoder model using beam search
        - beam_index: number of candidates to consider at every step
        '''
        hidden = self.decoder.reset_state(batch_size=1)
        features = self.encoder(img_tensor_val)
        dec_input = tf.expand_dims([self.tokenizer.word_index['startseq']], 0)
    
        max_len = self.max_len
        seq = [self.tokenizer.word_index['startseq']]
        #in_text -> [[sequence_till_now, prob, hidden, dec_input]]
        in_text = [
            [seq, 0.0, hidden, dec_input]
        ]
        endseq_id = self.tokenizer.word_index['endseq']
        finished_hypothesis = []

        while len(in_text) > 0:
            tempList = []
            for seq in in_text:
                this_hidden = seq[2]
                this_dec_input = seq[3]
                predictions, hidden, attention_weights = self.decoder(this_dec_input, features, this_hidden)
                predictions_log_prob = tf.math.log(tf.nn.softmax(predictions))
                top_pred_ids = tf.nn.top_k(predictions, beam_index)[1][0].numpy()
                for word_id in top_pred_ids:
                    next_seq, log_prob = seq[0][:], seq[1]
                    #stopping criterion -> if endseq encountered, but not as first prediction
                    if word_id == endseq_id:
                        if len(next_seq) == 1:
                            continue
                        finished_hypothesis.append([next_seq, log_prob/(len(next_seq)-1)])
                        continue
                    
                    next_seq.append(word_id)
                    log_prob += predictions_log_prob.numpy()[0][word_id]

                    #stopping criterion -> if max_len achieved
                    if len(next_seq) == max_len:
                        finished_hypothesis.append([next_seq, log_prob/(len(next_seq)-1)])
                        continue
                    
                    dec_input = tf.expand_dims([word_id], 0)
                    tempList.append([next_seq, log_prob, hidden, dec_input])
            in_text = tempList
            #sort according to probabilities
            in_text = sorted(in_text, reverse=True, key=lambda l: l[1])
            #take top words
            in_text = in_text[:beam_index]
            # print('done with current iteration, len of in_text =', len(in_text))
        
        # print('len of finished hypo =', len(finished_hypothesis))
        finished_hypothesis = sorted(finished_hypothesis, reverse=True, key=lambda l: l[1])
        pred_cap_ids = finished_hypothesis[0][0]
        pred = [self.tokenizer.index_word[x] for x in pred_cap_ids[1:]]
        pred = ' '.join(pred)
        # print('predicted_caption = {}\n'.format(pred))
        return pred
    
    def greedy_search_pred(self, img_tensor_val, compute_attention_plot=True, batch_size=1):
        '''
        predict captions from an already trained encoder-decoder model using greedy search. 
        greedy search considers the best probable word at each step
        '''
        attention_plot = np.zeros((self.max_len, 64))

        hidden = self.decoder.reset_state(batch_size=batch_size)        
        features = self.encoder(img_tensor_val)
        dec_input = tf.expand_dims([self.tokenizer.word_index['startseq']] * batch_size, 0)

        result = []
        for i in range(self.max_len):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)
            if compute_attention_plot:
                attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.nn.top_k(predictions, 1)[1][0][0].numpy()

            if predicted_id == 0 or self.tokenizer.index_word[predicted_id] == 'endseq':
                result = ' '.join(result)
                return result, attention_plot
            
            result.append(self.tokenizer.index_word[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)
        
        attention_plot = attention_plot[:len(result), :]
        result = ' '.join(result)
        return result, attention_plot
    
    def plot_attention(self, image, result, attention_plot):
        '''
        plot_attention <add documentation>
        '''
        temp_image = np.array(Image.open(image))
        
        fig = plt.figure(figsize=(10,10))
        len_result = len(result)
        for i in range(len_result):
            temp_att = np.resize(attention_plot[i], (8,8))
            grid_size = max(2, np.ceil(len_result/2))
            ax = fig.add_subplot(grid_size, grid_size, i+1)
            ax.set_title(result[i])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
        
        plt.tight_layout()
        plt.show()
    
    def compute_bleu_scores(self, config, group='val', search_method='greedy', filter_files=[]):
        '''
        generic function to compute BLEU scores, works with both prediction algorthims: greedy search and beam search
        computes BLEU scores for all images in train or val set, unless provided with list of files

        - config: config file 
        - group: 'val' or 'train' -> computes BLEU scores for all images in these sets
        - search_method: 'greedy' or 'beam'
        - filter_files: supply list of filenames in case you want to limit computation to a subset of train or val set
        '''
        print(f'{str(datetime.now())} computing BLEU scores for {group} set using {search_method} search ')
        df = get_image_to_caption_map(config['preprocessing'], group)
        if len(filter_files):
            df = df[df['filename'].isin(filter_files)]

        df['caption'] = df['caption'].apply(lambda x: ' '.join(x.split()[1:-1])) #remove startseq & endseq from captions
        filename_to_captions_dict = df.groupby('filename')['caption'].apply(list).to_dict()

        pred_df_rows = []
        for filename, captions_list in filename_to_captions_dict.items():
            img_tensor_val, _ = map_func(filename, '_' )
            if search_method == 'greedy':
                predicted_caption, _ = self.greedy_search_pred(img_tensor_val, compute_attention_plot=False)
            elif search_method == 'beam':
                predicted_caption = self.beam_search_pred(img_tensor_val)
            
            #compute BLEU scores
            captions_list_words = [str.lower(x[:-2]).split() for x in captions_list]
            predicted_caption_words = predicted_caption.split()
            bleu_1 = round(sentence_bleu(captions_list_words, predicted_caption_words, weights=[1.0, 0, 0, 0]), 3)
            bleu_2 = round(sentence_bleu(captions_list_words, predicted_caption_words, weights=[0.5, 0.5, 0, 0]), 3)
            bleu_3 = round(sentence_bleu(captions_list_words, predicted_caption_words, weights=[0.33, 0.33, 0.33, 0]), 3)
            bleu_4 = round(sentence_bleu(captions_list_words, predicted_caption_words, weights=[0.25, 0.25, 0.25, 0.25]), 3)
            # print(f'bleu_1 = {bleu_1}, bleu_2 = {bleu_2}, bleu_3 = {bleu_3}, bleu_4 = {bleu_4}')
            pred_df_rows.append([filename, bleu_1, bleu_2, bleu_3, bleu_4, predicted_caption] + captions_list)

        bleu_headers = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']
        caption_headers = ['pred_cap', 'real_cap_1', 'real_cap_2', 'real_cap_3', 'real_cap_4', 'real_cap_5']
        headers = ['filename' ] + bleu_headers + caption_headers
        pred_df = pd.DataFrame(pred_df_rows, columns=headers)
        print(pred_df.describe())

        #write to disk
        filename = config['store'] + config['nn_params']['pred_df_dir'] + 'result_{}_{}.csv'.format(group, search_method)
        pred_df.to_csv(filename, index=False)
        return pred_df

class CNN_Encoder(tf.keras.Model):
    ''''
    Encoder class, passes the img_tensors through a fully connected layer whose number of units is equal to the 
    embedding dimensions
    '''
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        #shape of x -> (bathch_size, attention_features_shape, features_shape) {(batch_size, 64, 2048) for inceptionV3}
        x = self.fc(x)
        #shape of x after fc -> (batch_size, attention_features_shape, embedding_dim)
        x = tf.nn.relu(x)
        return x
        
class BahdanauAttention(tf.keras.Model):
    '''
    add some literature about Bahdanau Attention
    '''
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, features, hidden):
        #features shape -> (batch_size, attention_features_shape, embedding_dim) -> this is the CNN_Encoder output
        #hidden shape -> (batch_size, hidden_size)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        #hidden_with_time_axis shape -> (batch_size, 1, hidden_size)
        
        attention_hidden_layer = (tf.nn.tanh((self.W1(features) + self.W2(hidden_with_time_axis))))
        #attention_hidden_layer shape -> (batch_size, 64, units)
        #why addition ? 

        score = self.V(attention_hidden_layer)
        #score shape -> (batch_size, attention_features_shape, 1) -> this gives us an unnormalized score for each image feature

        attention_weights = tf.nn.softmax(score, axis=1)
        #attention_weights shape -> (batch_size, attention_features_shape, 1)

        context_vector = attention_weights * features
        #context_vecotr shape -> (batch_size, attention_features_shape, embedding_dim )
        
        context_vector = tf.reduce_sum(context_vector, axis=1)
        #context_vector shape after sum -> (batch_size, embedding_dim)
        ####------------------------------------------------------------------
        #tensorflow code says (batch_size, hidden_size) but I don't think that is correct
        ####------------------------------------------------------------------
        
        return context_vector, attention_weights
    
class RNN_Decoder(tf.keras.Model):
    '''
    Decoder model that takes the encoded images as inputs, and runs them through a RNN to learn the sequence of captions
    '''
    def __init__(self, embedding_dim, units, vocab_size, embedding_matrix):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size, 
            output_dim = embedding_dim,
            weights = [embedding_matrix],
            trainable = False
        )
        self.gru1 = self.rnn_type()
        self.gru2 = self.rnn_type()

        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

        self.dropout = tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None)
        self.batchNormalization = tf.keras.layers.BatchNormalization()

    def rnn_type(self):
        # if tf.test.is_gpu_available():
        if len(tf.config.list_physical_devices('GPU')):
            return tf.compat.v1.keras.layers.CuDNNGRU(
                self.units, 
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )
        else:
            return tf.keras.layers.GRU(
                self.units, 
                return_sequences = True, 
                return_state = True, 
                recurrent_initializer = 'glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )

    def call(self, x, features, hidden):
        '''
        - 'features' are the encoded images features
        - 'hidden' is the hidden state of the RNN that is passed to predict next word in the sequence, initally initialized as zeros
        '''
        #get attention output
        context_vector, attention_weights = self.attention(features, hidden)
        #context_vector shape after sum -> (batch_size, embedding_dim)
        #attention_weights shape -> (batch_size, attention_features_shape, 1)

        x = self.embedding(x)
        #shape after embedding -> (batch_size, 1, embedding_dim)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        #shape after concatenation -> (batch_size, 1, embedding_dim + hidden_size)

        output, state = self.gru1(x)
        #shape of output = (batch_size, 1, units), shape of state = (batch_size, units)

        output, state = self.gru2(inputs=output, initial_state=state)
        #shape of output = (batch_size, 1, units), shape of state = (batch_size, units)

        x = self.fc1(output)
        #shape of x -> (batch_size, max_length, hidden_size) {max_length=1 because inputs are sent word by word}

        x = self.dropout(x)
        x = self.batchNormalization(x)

        x = tf.reshape(x, (-1, x.shape[2]))
        #shape of x -> (batch_size * max_length, hidden_size)

        x = self.fc2(x)
        #shape of x -> (batch_size * max_length, vocab_size)

        return x, state, attention_weights
    
    def build_graph(self):
        x = tf.keras.layers.Input(shape=(64,1))
        print('shape of x in build_graph =', x.shape)
        features = tf.zeros((64, 64, 300))
        hidden = tf.zeros((64,512))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, features, hidden))

    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
