import tensorflow as tf
tf.random.set_seed(1)
import numpy as np
import os
from utils.load_data_utils import load_config

def get_inceptionv3():
    print('\nentered into the func that loads inception v3 model\n')
    image_model = tf.keras.applications.InceptionV3(
        include_top=False, 
        weights='imagenet',
        # pooling ='avg',
        # input_shape = (299,299,3),
        # input_tensor=tf.keras.layers.Input(shape=(299, 299, 3)    
    )
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    # print('image model summary:\n', image_features_extract_model.summary())
    return image_features_extract_model

def extract_and_save_img_features(img_files, batch_size):
    print('\n-----received %d images to extract features...-----\n' % (len(img_files)))
    pretrained_model = get_inceptionv3()

    image_dataset = tf.data.Dataset.from_tensor_slices(img_files)
    image_dataset = image_dataset.map(
        load_image,     
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size)
    
    num_batches = len(img_files) // batch_size

    #get the encoded feature map for each image    
    numpy_paths = []
    curr_batch = 0    
    for img, path in image_dataset:
        curr_batch += 1
        print('entered batch number = %d out of %d batches' % (curr_batch, num_batches))
        # print('\nimg.shape in image_dataset iter = {}'.format(img.shape))
        batch_features = pretrained_model(img)
        
        # print('img.shape after getting features = {}'.format(img.shape))
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        # print('img.shape after reshape = {}\n'.format(img.shape))
        count = 0
        batch_paths = []
        for bf, path in zip(batch_features, path):
            path_ = path.numpy().decode('utf-8')

            config = load_config()['preprocessing']
            images_features_dir = config['images_features_dir']
            img_features_path = images_features_dir + path_.split('/')[-1]
            # print('img_features_path = ', img_features_path)

            np.save(img_features_path, bf.numpy())   
            numpy_paths.append(img_features_path)       
            count += 1
        # print('num iterations over image save for this batch = {}\n'.format(count))
    
    # #write numpy_paths to disk
    # with open('data/numpy_paths_numfiles{}'.format(len(img_files)), 'w') as file_handler:
    #     for item in numpy_paths:
    #         file_handler.write('{}\n'.format(item))

    return image_dataset
        
def load_image(filename):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, filename

def map_func(filename, cap, full_path=True):
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8')
    config = load_config()['preprocessing']
    images_features_dir = config['images_features_dir']
    if full_path:
        img_features_path = images_features_dir + filename.split('/')[-1] + '.npy'
    else:
        img_features_path = images_features_dir + filename + '.npy'
    img_tensor = np.load(img_features_path)
    return img_tensor, cap

def load_image_features_from_disk(img_files_list, cap_padded_list, BUFFER_SIZE, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((img_files_list, cap_padded_list))
    dataset = dataset.map(
        lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def load_image_features_on_the_fly(filename):
    config = load_config()
    short_filename = filename.split('/')[-1]
    img_features_filename = config['preprocessing']['images_features_dir'] + short_filename + '.npy'
    if os.path.exists(img_features_filename):
        img_tensor_val, _ = map_func(short_filename, '', full_path=False)
    else:
        pretrained_model = get_inceptionv3()
        temp_input = tf.expand_dims(load_image(filename)[0], 0)    
        print('shape of temp_input = ', temp_input.shape)
        img_tensor_val = pretrained_model(temp_input)
        print('shape of img_tensor_val = ', img_tensor_val.shape)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    
    return img_tensor_val