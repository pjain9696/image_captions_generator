import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import yaml, json

def load_config():
    with open('./config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()
    return config

def get_karpathy_split():
    '''
    load karpathy split available as a json file in directory
    '''
    config = load_config()
    karpathy_split_dir = config['preprocessing']['karpathy_test_val_split_json']

    with open(karpathy_split_dir) as f:
        karpathy_split = json.load(f)['images']
        f.close()
    return karpathy_split

def get_image_data(config, group, data_label):
    '''
    group: train / val / test
    data_label: imgid / filename 
    '''
    karpathy_json = get_karpathy_split()
    item = [x[data_label] for x in karpathy_json if x['split'] == group]

    if data_label == 'filename':
        images_dir = config['images_dir']
        #add file path to file name in order to make it accessible
        item = [images_dir + x for x in item]
    return item

def get_image_to_caption_map(config, group):
    '''
    group = train / val / test
    obtain a dataframe containing image file names and their human annotated captions
    '''
    karpathy_split = get_karpathy_split()
    karpathy_subset = [x for x in karpathy_split if x['split'] == group]
    
    images_list = []
    captions_list = []
    images_dir = config['images_dir']

    for item in karpathy_subset:
        filename = images_dir + item['filename']
        captions = [x['raw'] for x in item['sentences']]
        images_list += [filename]*len(captions)
        captions_list += captions
    
    ret_df = pd.DataFrame({'filename':images_list, 'caption':captions_list})

    #add start token 'startseq' and end token 'endseq' to each caption
    ret_df['caption'] = ret_df['caption'].apply(lambda x: 'startseq ' + x + ' endseq')

    if config.get('experiment_size', None) is not None:
        ret_df = ret_df[:config['experiment_size']]    
    
    return ret_df

def get_all_captions_for_image(image_filename):
    '''
    provided a single filename, return all 5 human annotated captions
    '''
    karpathy_split = get_karpathy_split()
    img_data = [item for item in karpathy_split if item['filename'] == image_filename]
    caption_list = []
    for sentence in img_data:
        caption_list += [item['raw'] for item in sentence['sentences']]
    return caption_list
    
def load_pretrained_embeddings(file_path):
    '''
    load Glove pretrained word vector embeddings from disk
    '''
    embeddings_index = {}
    print('\nembed file path = {}\n'.format(file_path))

    f = open(file_path)
    for line in tqdm(f):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("Found %s word vectors" % len(embeddings_index))
    return embeddings_index

def get_tokenizer_from_dir(config):
    ''''
    load the tokenizer file saved as json in disk. Used to convert words to tokens and vice-versa.
    '''
    #load tokenizer
    tokenizer_dir = config['preprocessing']['tokenizer_dir']
    with open(tokenizer_dir) as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    return tokenizer

def plot_losses(filename):
    '''
    helper function to plot train and val losses per epochs
    '''
    loss_df = result = pd.read_csv(filename)

    plt.figure(figsize=(7,4))
    plt.plot(range(1,len(loss_df)), loss_df['train_loss'], label='train_loss')
    plt.plot(range(1,len(loss_df)), loss_df['val_loss'], label='val_loss')
    plt.xticks(range(1,len(loss_df)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Per Epoch')
    plt.grid()
    plt.legend()
    plt.rcParams['figure.facecolor'] = 'white'
    plt.savefig('./data/loss_per_epoch_plot.png')
    plt.show()
    return        