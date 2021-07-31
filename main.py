import os
from module.preprocessor import Preprocessor
from module.trainer import Trainer
from utils.load_data_utils import load_config

if __name__ == '__main__':
    config = load_config()

    if not os.path.exists(config['store']):
        os.mkdir(config['store'])
        os.mkdir(config['store'] + config['nn_params']['pred_df_dir'])

    #preprocessing: obtain training / validation / test data
    pp = Preprocessor(config)
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
    trainer.initiate_training(
        train_dataset, 
        val_dataset, 
        load_from_checkpoint=True, 
        load_loss_file=True, 
        save_loss_to_dir=True,
    )

    # metrics
    bleu_df = trainer.compute_bleu_scores(config, group='val', search_method='greedy')
    bleu_df = trainer.compute_bleu_scores(config, group='train', search_method='greedy')
    bleu_df = trainer.compute_bleu_scores(config, group='val', search_method='beam')
    # bleu_df = trainer.compute_bleu_scores(config, group='train', search_method='beam')

    # filename_short = './data/Flicker8k_Images/3695064885_a6922f06b2.jpg'
    # pred_cap = trainer.compute_bleu_scores(config, 'val', filter_files=[filename_short])
    # print(pred_cap)