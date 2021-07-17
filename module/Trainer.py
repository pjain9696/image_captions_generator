# from re import A
from extract_img_features import load_image
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from utils import get_tokenizer_from_dir
from extract_img_features import load_image, load_image_features_on_the_fly

class Trainer:
    def __init__(self, config, vocab_size, max_len, embedding_matrix):
        self.config = config['nn_params']
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
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.config['checkpoint_dir'], max_to_keep=5)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype = loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0

        #initializing the hidden state for each batch, because the captions are not related image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        #shape of hidden -> (batch_size, units)

        dec_input = tf.expand_dims([self.tokenizer.word_index['startseq']] * target.shape[0], 1)
        # print('shape of de_input =', dec_input.shape)
        #shape of dec_input -> (batch_size, 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)
            #shape of features after coming out of encoder fc -> (batch_size, 64, embedding_dim)

            for i in range(1, target.shape[1]):
                #passing the features through decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)

                # print('predictions shape =', predictions.shape)
                # print('target[:,i] =', tf.print(target[:,i]))
                loss += self.loss_function(target[:,i], predictions)

                #using teacher forcing
                dec_input = tf.expand_dims(target[:,i], 1)
        
        total_loss = (loss / int(target.shape[1]))
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def initiate_training(self, train_dataset):
        EPOCHS = self.config['EPOCHS']
        loss_plot = []
        for epoch in range(EPOCHS):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                # print('entered batch number = ', batch)
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                    # print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
            
            #storing the epoch end loss value to plot later
            num_steps = train_dataset.cardinality().numpy()
            avg_loss_this_epoch = total_loss / num_steps
            loss_plot.append(avg_loss_this_epoch)

            #save the epoch with the lowest loss
            if (epoch+1) % 5 == 0:
                print('saving a checkpoint, epoch = ', epoch+1)
                self.checkpoint_manager.save()
            
            print(f'Epoch {epoch+1} Average Loss {avg_loss_this_epoch:.6f}, time taken {time.time()-start:.2f}sec \n')
        
        #plot the losses
        plt.plot(loss_plot)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Plot')
        plt.grid()
        plt.show()
        return

    def evaluate(self, image):
        attention_plot = np.zeros((self.max_len, 64))
        hidden = self.decoder.reset_state(batch_szie=1)

        img_tensor_val = load_image_features_on_the_fly(image)
        features = self.encoder(img_tensor_val)
        dec_input = tf.expand_dims([self.tokenizer.word_index['startseq']], 0)

        result = []
        for i in range(self.max_len):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)
            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            print('predictions = ', predictions)
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == 'endseq':
                return result, attention_plot
            
            dec_input = tf.expand_dims([predicted_id], 0)
        
        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot
    
    def plot_attention(self, image, result, attention_plot):
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

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        #shape after fc -> (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
        
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, features, hidden):
        #features (CNN_Encoder output) shape -> (batch_size, 64, embedding_dim)
        #hidden shape -> (batch_size, hidden_size)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        #hidden_with_time_axis shape -> (batch_size, 1, hidden_size)
        
        attention_hidden_layer = (tf.nn.tanh((self.W1(features) + self.W2(hidden_with_time_axis))))
        #attention_hidden_layer shape -> (batch_size, 64, units)
        #why addition ? 

        score = self.V(attention_hidden_layer)
        #score shape -> (batch_size, 64, 1) :: this gives us an unnormalized score for each image feature

        attention_weights = tf.nn.softmax(score, axis=1)
        #attention_weights shape -> (batch_size, 64, 1)

        context_vector = attention_weights * features
        # print('shape of context_vector = ', context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # print('shape of context_vector after sum = ', context_vector.shape)
        #context_vector shape after sum -> (batch_size, hidden_size) or (batch_size, embedding_dim) ?

        return context_vector, attention_weights
    
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, embedding_matrix):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size, 
            output_dim = embedding_dim,
            weights = [embedding_matrix],
            trainable = False
        )
        self.gru = tf.keras.layers.GRU(
            self.units, 
            return_sequences = True, 
            return_state = True, 
            recurrent_initializer = 'glorot_uniform'
        )
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)
    
    def call(self, x, features, hidden):
        #get attention output
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)
        #shape after embedding -> (batch_size, 1, embedding_dim)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        #shape after concatenation -> (batch_size, 1, embedding_dim + hidden_size)
        # print('shape of x after concat =', x.shape)

        output, state = self.gru(x)
        # print('shape of output =', output.shape)
        # print('shape of state =', state.shape)

        x = self.fc1(output)
        #shape of x -> (batch_size, max_length, hidden_size)
        # print('shape of x after fc1 =', x.shape)

        x = tf.reshape(x, (-1, x.shape[2]))
        #shape of x -> (batch_size * max_length, hidden_size)
        # print('shape of x after reshape =', x.shape)

        x = self.fc2(x)
        #shape of x -> (batch_size * max_length, vocab_size)
        # print('shape of x after fc2 =', x.shape)

        return x, state, attention_weights
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))



