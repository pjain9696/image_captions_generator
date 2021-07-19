from module.Preprocessor import Preprocessor
from module.Trainer import Trainer
from utils import load_config, get_all_captions_for_image

# config = load_config()
# print(config)
if __name__ == '__main__':
    config = load_config()

    #preprocessing: obtain training / validation / test data
    pp = Preprocessor(config)
    # val = pp.prep_data()
    train_dataset, val_dataset = pp.prep_data()

    print('shape of embedding_matrix =', pp.embedding_matrix.shape)
    print('vocab_size = ', pp.vocab_size) 
    print('cardinality of train_dataset = ', train_dataset.cardinality().numpy())
    print('cardinality of val_dataset = ', val_dataset.cardinality().numpy())

    for element in train_dataset:
        print('shape of elements = {}, {}'.format(element[0].shape, element[1].shape))
        break
     
    print('-'*100)
    
    #training
    trainer = Trainer(config, pp.vocab_size, pp.max_len, pp.embedding_matrix)
    trainer.initiate_training(train_dataset, val_dataset, load_from_checkpoint=False)

    #check the output on a sample test image
    image = './data/Flicker8k_Images/3695064885_a6922f06b2.jpg'
    image_rawname = image.split('/')[-1]
    result, attention_plot = trainer.evaluate(image)
    print('predicted caption is = {}\n'.format(result))

    real_captions = get_all_captions_for_image(image_rawname)
    print('\nreal captions = {}\n'.format(real_captions))
    trainer.plot_attention(image, result, attention_plot)

    #train
    '''specify the training algorithm to use'''
    '''pass in train/val sets to training algorithm'''