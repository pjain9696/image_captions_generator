import pandas as pd
from datetime import datetime as dt

def load_captions(filename):
    '''create a dict of image to their captions:
    {
        image_id1 : [caption1, caption2,...]
        image_id2 : [caption1, caption2,...]
    }
    '''
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    
    captions_dict = dict()
    _count = 0
    for line in doc.split('\n'):
        if len(line) < 2:
            continue
        #split line on whitespaces
        tokens = line.split()
        image_id, image_caption = tokens[0].split('#')[0], tokens[1:]
        image_caption = ' '.join(image_caption)
        
        if image_id not in captions_dict:
            captions_dict[image_id] = list()
        captions_dict[image_id].append(image_caption)
        _count += 1
    print('{}: Parsed captions: {}'.format(dt.now(), _count))
    return captions_dict

def load_data(captions_file, images_folder):
    file = open(captions_file, 'r')
    doc = file.read()
    file.close()

    data = []
    for line in doc.split('\n'):
        line = line.split('\t')
        if len(line) < 2:
            continue
        row = line[0].split('#') + [line[1].lower()]
        data.append(row)
    
    data = pd.DataFrame(data, columns=['filename', 'index', 'caption'])
    data = data[['index', 'filename', 'caption']]
    return data





