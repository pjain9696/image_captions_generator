import os
from utils.load_data_utils import load_config, get_image_to_caption_map
from utils.img_features_utils import extract_and_save_img_features

def main():
    config = load_config()

    train_df = get_image_to_caption_map(config['preprocessing'], 'train')
    val_df = get_image_to_caption_map(config['preprocessing'], 'val')

    train_img_files = train_df['filename'].tolist()    
    val_img_files = val_df['filename'].tolist()
    all_files = train_img_files + val_img_files
    
    print('len of train_img_files = ', len(train_img_files))
    print('len of val_img_files = ', len(val_img_files))
    print('len of all_files = ', len(all_files))

    #extract image features using transfer learning if not done already (or during the first run!)
    if not os.path.exists(config['preprocessing']['images_features_dir']):
        extract_and_save_img_features(all_files, config['nn_params']['BATCH_SIZE'])   

if __name__ == '__main__':
    '''
    extract image features from raw images using a pretrained CNN model and save to disk
    '''
    main()