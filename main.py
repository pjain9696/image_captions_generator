from module.Preprocessor import Preprocessor
from module.Trainer import Trainer
from utils import load_config

if __name__ == '__main__':
    config = load_config()

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
        save_loss_to_dir=True
    )

    #metrics
    # bleu_df = trainer.compute_bleu_scores(config, group='val', search_method='greedy')
    # bleu_df = trainer.compute_bleu_scores(config, group='val', search_method='beam')

    # filename_short = '3695064885_a6922f06b2.jpg'
    # pred_cap = trainer.predict_caption_beam_search(filename_short)
    # print(pred_cap)

    #todo:
    # try other pretrained models instead of Inception v3