import os
import json 
import cv2
import h5py
import numpy as np
import configparser
import pandas as pd
from tqdm import tqdm   
from collections import Counter
from random import choice, sample
from nltk.tokenize import word_tokenize

#the function to read the json file (annotations of COCO dataset)
def read_file(file_path):
    with open(file_path, 'r') as f:
        COCO = json.load(f)
    return COCO

#the function to save the json file
def save_json(json_data, file_path):
    with open(file_path, 'w') as f:
        f.write(json.dumps(json_data))

#the function to tokenize the sentence into words
def tokenize_captions(caption):
    tokens = word_tokenize(caption.lower())
    tokens = [word for word in tokens if word.isalpha()]
    return tokens


#the function makes the structure of the main dataset, the structure is a dictionary that contains the images and their captionsand details
def make_json_structure(info, dataset_path):
    rows = []
    for key, value in info.items():
        COCO = read_file(value)
        
        COCO_df = pd.DataFrame(COCO["annotations"])
        images = pd.DataFrame(COCO["images"])
        image_id_to_filename = dict(zip(images['id'], images['file_name']))
        
        for image_id, group in COCO_df.groupby("image_id"):
            print(image_id, key)
            row = {}
            row["filename"] = image_id_to_filename[image_id] 
            row["sentences_ids"] = [id for id in group["id"]]
            row["split"] = key
            row["image_id"] = image_id
            row["sentences"] = []
            for _, annotation in group.iterrows():
                sentence = {
                    "tokens": tokenize_captions(annotation["caption"]),
                    "raw": annotation["caption"],
                    "sentence_id" : annotation["id"],
                    "img_id" : image_id,
                }
                row["sentences"].append(sentence)
            
            rows.append(row)

    print("number of pictures inside the dataset: ", len(rows))
    print("number of captions inside the dataset: ", sum([len(row["sentences"]) for row in rows]))
    res = {"images" : rows}
    save_json(res, dataset_path)




#the function to create the files for the training and validation of the model, files will contain the images, the captions, and the word map
def create_files(word_map_path, json_dataset_path, image_folder, base_path, 
                       captions_for_image, min_word_freq, max_len=100):

    with open(json_dataset_path, 'r') as j:
        data = json.load(j)

    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        if img['split'] in {'train'}:
            train_image_paths.append(os.path.join(image_folder, "train2017", img['filename']))
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(os.path.join(image_folder, "val2017", img['filename']))
            val_image_captions.append(captions)

    #check if the lists have the same len to avoid any errors
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)

    # create a word map that assigns a unique index to each word in the vocabulary
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    print("Vocabulary size: {}".format(len(words)))
    # Save word map to a JSON
    save_json(word_map, word_map_path)
    print("Word map saved to: {}".format(word_map_path))
   
    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                ]:
        

        with h5py.File(os.path.join(base_path, "IMAGES_" + split + '.hdf5'), 'a') as h:
            
            h.attrs['captions_for_image'] = captions_for_image
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("Reading %s images and captions, storing to file" % split)

            encoded_captions = []
            caption_lens = []

            for i, path in enumerate(tqdm(impaths)):
                if len(imcaps[i]) < captions_for_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_for_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_for_image)

                assert len(captions) == captions_for_image

                img = cv2.imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = cv2.resize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                images[i] = img

                for j, c in enumerate(captions):
                    encoded_caption = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    caption_len = len(c) + 2

                    encoded_captions.append(encoded_caption)
                    caption_lens.append(caption_len)

            assert images.shape[0] * captions_for_image == len(encoded_captions) == len(caption_lens)

            # Save encoded captions and their lengths to JSON files
            save_json(encoded_captions, os.path.join(base_path, "Encoded_Captions_" + split + '.json'))
            save_json(caption_lens, os.path.join(base_path, "Captions_len_" + split + '.json'))



if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini')
    coco_caption_val_path = config['coco_path']['val_captions']
    coco_caption_train_path = config['coco_path']['train_captions']
    dataset_path = config['json_path']['json_dataset']
    word_map_path = config['json_path']['word_map_path']
    image_folder = config['coco_path']['image_folder']
    base_path = config['json_path']['base_path_json']

    info = {
        "val" : coco_caption_val_path,
        "train" : coco_caption_train_path,
    }

    make_json_structure(info, dataset_path)
    create_files(word_map_path, dataset_path, image_folder, base_path, 5, 3)

