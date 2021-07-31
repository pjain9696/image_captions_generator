import pandas as pd
import numpy as np
import tensorflow as tf
tf.random.set_seed(1)
import os, io, json
from utils.load_data_utils import load_config, get_image_to_caption_map, load_pretrained_embeddings
from utils.img_features_utils import get_inceptionv3, extract_and_save_img_features, load_image_features_from_disk
class Preprocessor:
    def __init__(self, config) -> None:
        self.config = config['preprocessing']
        self.nn_params = config['nn_params']
        self.store_dir = config['store']
        self.train_df = get_image_to_caption_map(self.config, 'train')
        self.val_df = get_image_to_caption_map(self.config, 'val')
        self.test_df = get_image_to_caption_map(self.config, 'test')
        self.tokenizer = None #set in func ___
        self.vocab_size = None #set in func ___
        self.max_len = None #set in func ___
        self.embedding_matrix = None # set in func ___
    
    def prep_data(self):
        '''
        #1. tokenize the captions; determine vocab size / max_len;  get the embedding matrix
        '''
        train_cap_padded, val_cap_padded, tokenizer, self.vocab_size, self.max_len = self.extract_caption_features()        
        self.embedding_matrix = self.get_embedding_matrix(tokenizer)
        
        '''
        #2. get the encoded features from the images
        '''
        train_img_files = self.train_df['filename'].tolist()
        val_img_files = self.val_df['filename'].tolist()
        all_files = train_img_files + val_img_files
        
        print('len of train_img_files = ', len(train_img_files))
        print('len of val_img_files = ', len(val_img_files))
        print('len of all_files = ', len(all_files))

        #extract image features using transfer learning if not done already (or during the first run!)
        if not os.path.exists(self.config['images_features_dir']):
            extract_and_save_img_features(all_files, self.nn_params['BATCH_SIZE'])        
       
        train_dataset = load_image_features_from_disk(train_img_files, train_cap_padded, self.nn_params['BUFFER_SIZE'], self.nn_params['BATCH_SIZE'])
        val_dataset = load_image_features_from_disk(val_img_files, val_cap_padded, self.nn_params['BUFFER_SIZE'], self.nn_params['BATCH_SIZE'])
        return train_dataset, val_dataset

    def extract_caption_features(self):
        train_captions = self.train_df['caption'].tolist()
        val_captions = self.val_df['caption'].tolist()
        
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.nn_params['vocab_size'])
        tokenizer.fit_on_texts(train_captions)
        
        train_cap_tokenized = tokenizer.texts_to_sequences(train_captions)
        val_cap_tokenized = tokenizer.texts_to_sequences(val_captions)
        
        #set the vocabulary size based on unique words present in train set
        vocab_size = min(self.nn_params['vocab_size'], len(tokenizer.word_index))
        print('vocab_size = ', vocab_size)

        #figure out the length of captions in train set
        max_len = max([len(x) for x in train_cap_tokenized])
        print('max_len = ', max_len)

        train_cap_padded = tf.keras.preprocessing.sequence.pad_sequences(train_cap_tokenized, maxlen=max_len, padding='post')
        val_cap_padded = tf.keras.preprocessing.sequence.pad_sequences(val_cap_tokenized, maxlen=max_len, padding='post')

        #save tokenizer to disk
        with io.open(self.config['tokenizer_dir'], 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
        
        return train_cap_padded, val_cap_padded, tokenizer, vocab_size, max_len
    
    def get_embedding_matrix(self, tokenizer):
        filtered_embedding_dir = './data/filtered_embed_vocabsize{}.csv'.format(self.vocab_size)

        #get the embedding matrix
        if os.path.exists(filtered_embedding_dir):
            print('\nloading pre-saved filtered embedding from disk, filename = {}...\n'.format(filtered_embedding_dir))
            embedding_matrix = np.genfromtxt(filtered_embedding_dir, delimiter=',')
        else:
            print('creating an embedding_matrix of shape = ({},{})\n'.format(self.vocab_size, self.nn_params['embedding_dim']))
            embeddings_index = load_pretrained_embeddings(self.config['raw_pretrained_embedding_dir'])
            #initialize embedding_matrix with default zero values
            embedding_matrix = np.zeros((self.vocab_size, self.nn_params['embedding_dim']))

            oov_words = []
            for word, i in tokenizer.word_index.items():
                if i >= self.vocab_size:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                else:
                    oov_words.append(word)
            
            #save the oov_words to disk to analyse them later
            oov_words_dir = './data/oov_words_vocabsize{}'.format(self.vocab_size)
            with open(oov_words_dir, 'w') as file_handler:
                for item in oov_words:
                    file_handler.write('{}\n'.format(item))
            print('out of %d words in vocab, %s are missing from glove vocab\n' % (self.vocab_size, len(oov_words)))

            #save embedding_matrix to disk for effective use of RAM in subsequent runs 
            np.savetxt(filtered_embedding_dir, embedding_matrix, delimiter=',')
        return embedding_matrix