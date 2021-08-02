# Image Captions Generator

Generate captions from images<br>
{readme update work in progress, keep tuning in!}

## Table Of Contents

1. [Background](#1-background)
    * 1.1. [Why caption images](#11-why-caption-images)
    * 1.2. [Model architecture](#12-model-architecture)
    * 1.3. [Why use attention](#13-why-use-attention)
2. [How to use this repository](#2-how-to-use-this-repository)
    * 2.1. [Requirements](#21-requirements)
    * 2.2. [Configure settings](#22-configure-settings)
    * 2.3. Create
    * 2.4. Training
    * 2.5. Inference 
3. Results
4. Inferences
5. References

## 1. Background

### 1.1. Why caption images

Image captions serve as a link between the image and the text. It helps AI algortihms to make better sense of what's going on in the image. Can be useful for Seach Engine Optimization (SEO), indexing and archiving photos based on its contents such as actions / places / objects in it (similar to what Google Photos does today). Image captioning can also help to process videographic data by identifying what's happening frame by frame.

### 1.2. Model architecture

Here we are using an attention based encoder-decooder model to generate captions. Training images available in RBG format are passed through a pre-trained encoder model to obtain spatial information from images, which are then passed through a decoder block to generate the captions sequentially. Encoder is most oftenly a pre-trained CNN model and decoder is a RNN model. 

### 1.3. Why use attention ?

Attention in literal English means directing focus at something or taking greater notice. In Deep Learning, attention mechanism lives off the same concept where a model pays higher focus on certain factors while processing the data. Attention in an encoder-decoder model is in charge of managing and quantifying the dependence that the decoder has on the encoder.

## 2. How to use this repository

### 2.1. Requirements

* Create a conda environment using following command:<br>
`conda create --name <_yourenvname_> python=3.7`
In case you are not familiar with conda environments, refer to [[1]](#1-getting-started-with-python-environments-using-conda) and [[2]](#2-create-virtual-environments-for-python-with-conda) to get started.

* Install dependencies using [requirements.txt](requirements.txt)<br>
`pip3 install -r requirements.txt`

* Flickr8k images have been used in this project. There are multiple sources available online to download this dataset, here is one of them - [download flickr8k](https://www.kaggle.com/adityajn105/flickr8k).

* Instead of creating the train-test-val split by ourselves, we'll leverage the awesome work done by Andrej Karpathy to split the images into train-val-test in ratio  6000:1000:1000 alongwith segregating their captions. This is available as json, download from [here](https://www.kaggle.com/shtvkumar/karpathy-splits?select=dataset_flickr8k.json).
    
### 2.2. Configure settings

Specify following settings in [config.yaml](config/config.yaml). These are used as hyperparameters to train the model.<br>

* **epochs** Number of cycles where all training data is passed through the model exactly once. A forward pass and a backward pass together are counted as one pass<br>
* **batch_size:** An epoch is made up of one or more batches, where we use a part of the dataset to train the neural network. Training data is split into batch_size and model weights are updated once per batch<br>
* **units:** Number of units in the decoder RNN<br>
* **embedding_dim:**<br>
* **vocab_size:**<br>

### 2.3. Training and Inference

* In the preprocessing step, tensorflow.keras.preprocessing.pad_sequences is used to standardize the length of all training captions, we have set max_len parameter of pad_sequences to 21 , which is equal to {average(training captions) + 2 standard deviation(training captions)}. Keeping max_len = 39 (longest caption in train) leads to spurious results where the algorithm lurks between random words until max_len is reached. 

Also, I figured that only 2.2% training captions are longer than 21 words, so it does not make sense to train for full size.

<p align='center' width='100%'>
    <img src='data/plots/caption_length_distribution.png'>
</p>

* Run [img_features.py](img_features.py) to save the encoded version of images into the disk. This usually takes 30-35 mins for images in train and validation set, so better to perform this step one-off before kicking off training. Saving to disk would save us from encoding images repeatedly as we try different combinations of model settings.

* At this point, we are ready with all settings to begin the model training. Kick-off training by running [main.py](main.py).

* At the end of training, we compute BLEU scores for both training and validation set, using two methods to generate predicted captions: Greedy Search and Beam Search

* Here are the losses per epoch for training and validation sets. While the training loss decreases as training progresses, there is a hint of overfitting as the validation loss increases after an initial dip.

<p align='center' width='100%'>
    <img src='data/plots/loss_per_epoch_plot.png'>
</p>

## 3. Results

| # 	| max_len 	|                      Model Setting                      	|                                               Greedy Search                                              	|                                             Beam Search (k=3)                                            	|
|---	|---------	|:-------------------------------------------------------:	|:--------------------------------------------------------------------------------------------------------:	|:--------------------------------------------------------------------------------------------------------:	|
| 1 	| 39      	| Epochs: 40<br>RNN Units: 256<br>Vocab size: Full (7378) 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4044<br>- BLEU-2: 0.2131<br>- BLEU-3: 0.0792<br>- BLEU-4: 0.0234 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4259<br>- BLEU-2: 0.2284<br>- BLEU-3: 0.0962<br>- BLEU-4: 0.0358 	|
| 2 	| 39      	| Epochs: 20<br>RNN Units: 512<br>Vocab size: Full (7378) 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4548<br>- BLEU-2: 0.2560<br>- BLEU-3: 0.1183<br>- BLEU-4: 0.0400 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4547<br>- BLEU-2: 0.2711<br>- BLEU-3: 0.1301<br>- BLEU-4: 0.0466 	|
| 3 	| 39      	| Epochs: 40<br>RNN Units: 512<br>Vocab size: 6000        	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4280<br>- BLEU-2: 0.2275<br>- BLEU-3: 0.0823<br>- BLEU-4: 0.0241 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4255<br>- BLEU-2: 0.2261<br>- BLEU-3: 0.0888<br>- BLEU-4: 0.0290 	|
| 4 	| 39      	| Epochs: 20<br>RNN Units: 512<br>Vocab size: 6000        	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4531<br>- BLEU-2: 0.2581<br>- BLEU-3: 0.1022<br>- BLEU-4: 0.0326 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4620<br>- BLEU-2: 0.2646<br>- BLEU-3: 0.1157<br>- BLEU-4: 0.0393 	|
| 5 	| 21      	| Epochs: 20<br>RNN Units: 512<br>Vocab size: 7377        	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4604<br>- BLEU-2: 0.2597<br>- BLEU-3: 0.1083<br>- BLEU-4: 0.0405 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4647<br>- BLEU-2: 0.2724<br>- BLEU-3: 0.1169<br>- BLEU-4: 0.0452 	|
| 6 	| 21      	| Epochs: 20<br>RNN Units: 512<br>Vocab size: 6000        	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4648<br>- BLEU-2: 0.2649<br>- BLEU-3: 0.1109<br>- BLEU-4: 0.0373 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4732<br>- BLEU-2: 0.2731<br>- BLEU-3: 0.1258<br>- BLEU-4: 0.0447 	|
| 7 	| 21      	| Epochs: 20<br>RNN Units: 256<br>Vocab Size:             	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4601<br>- BLEU-2: 0.2577<br>- BLEU-3: 0.1014<br>- BLEU-4: 0.0319 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4726<br>- BLEU-2: 0.2680<br>- BLEU-3: 0.1161<br>- BLEU-4: 0.0386 	|
| 8 	| 21      	| Epochs: 20<br>RNN Units: 512<br>Vocab Size: 7378        	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4745<br>- BLEU-2: 0.2776<br>- BLEU-3: 0.1219<br>- BLEU-4: 0.0398 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4855<br>- BLEU-2: 0.2900<br>- BLEU-3: 0.1359<br>- BLEU-4: 0.0523 	|
| 9 	| 17      	| Epochs: 20<br>RNN Units: 512<br>Vocab Size: 7378        	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4706<br>- BLEU-2: 0.2678<br>- BLEU-3: 0.1114<br>- BLEU-4: 0.0339 	| Dev set BLEU scores:<br><br>- BLEU-1: 0.4776<br>- BLEU-2: 0.2739<br>- BLEU-3: 0.1232<br>- BLEU-4: 0.0412 	|

* Here is how the model learnt captions for a sample image shown below:

| Image 	|                                                                                                                                                                                                                                                                                                                                                                                Intermediate Predicted Caption at each Epoch                                                                                                                                                                                                                                                                                                                                                                                	|
|:-----:	|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| ![biker](data/110595925_f3395c8bd6.jpg)     	| 1,a man on a bike on the street<br>2,a man on a bicycle<br>3,a man is riding a bicycle<br>4,a man is riding a bicycle down a street<br>5,a man on a bicycle<br>6,a man walks along a dirt road<br>7,a man is riding a bicycle down a road<br>8,a bicycle on a dirt bike<br>9,a man is driving a bicycle down a road<br>10,a man is riding a bicycle on a road<br>11,a man is riding his bike<br>12,a bicycle on a bicycle down a street<br>13,a bicycle wheel on a gravel road<br>14,a bicycle wheel bikes on a dirt road<br>15,a bicycle on a dirt road<br>16,a man is riding his bike covered road in a race<br>17,a man bicycle wheel<br>18,a bicycle wheel a bicycle moving down a curved pipe wall<br>19,a bicyclist on a curved road in the background<br>20,a man is riding a bicycle down a street 	|
|       	|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
## 5. Inferences
* Training on a larger corpus of images could potentially improve performance. Since I have used Google Colab for training and there is a cap on Google Drive storage limit, I have used Flickr8k dataset, but we could upgrade to Flickr30k or COCO dataset for access to more training samples.

* Training for higher epochs cause the model to overfit on the training set. We have incorporated regularization techniques like L2 regularizer and Dropout but they do not seem to help after a certain degree.

* BLEU scores are better while using Beam Search as compared to Greedy Search. Increasing the beam width (currently 3) can further improve the performance marginally. However, considering more beam samples at every step would mean higher computation expense.

* Increasing the number of units in RNN Decoder helps in better performance.

## 5. References

### [1. Getting started with Python environments using Conda](https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307)
### [2. Create virtual environments for python with conda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)


