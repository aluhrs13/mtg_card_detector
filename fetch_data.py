import ast
import json
import os
import pandas as pd
import re
from urllib import request, error
import imagehash as ih
from PIL import Image
import cv2
import numpy as np

from config import Config

"""
Note: All codes in this file realies on Scryfall API to aggregate card database and their images.
Scryfall API doc is available at: https://scryfall.com/docs/api
"""


def calc_image_hashes(card_pool, save_to=None, hash_size=None):
    """
    Calculate perceptual hash (pHash) value for each cards in the database, then store them if needed
    :param card_pool: pandas dataframe containing all card information
    :param save_to: path for the pickle file to be saved
    :param hash_size: param for pHash algorithm
    :return: pandas dataframe
    """
    if hash_size is None:
        hash_size = [16, 32]
    elif isinstance(hash_size, int):
        hash_size = [hash_size]
    
    # Since some double-faced cards may result in two different cards, create a new dataframe to store the result
    new_pool = pd.DataFrame(columns=list(card_pool.columns.values))
    for hs in hash_size:
            new_pool['card_hash_%d' % hs] = np.NaN
    for ind, card_info in card_pool.iterrows():
        if ind % 100 == 0:
            print('Calculating hashes: %dth card' % ind)

        card_names = []
        # Double-faced cards have a different json format than normal cards
        if card_info['layout'] in ['transform', 'double_faced_token', 'modal_dfc']:
            if isinstance(card_info['card_faces'], str):
                card_faces = ast.literal_eval(card_info['card_faces'])
            else:
                card_faces = card_info['card_faces']
            for i in range(len(card_faces)):
                card_names.append(card_faces[i]['name'])
        else:  # if card_info['layout'] == 'normal':
            card_names.append(card_info['name'])

        for card_name in card_names:
            # Fetch the image - name can be found based on the card's information
            card_info['name'] = card_name
            img_name = '%s/imgs/png/%s.png' % (Config.data_dir, card_info['id'])
            card_img = cv2.imread(img_name)

            # If the image doesn't exist, download it from the URL
            if card_img is None:
                print('WARNING: card %s is not found!' % img_name)  
                continue
                fetch_card_image(card_info, out_dir='%s/card_img/png' % (Config.data_dir))
                card_img = cv2.imread(img_name)

            # Compute value of the card's perceptual hash, then store it to the database
            img_card = Image.fromarray(card_img)
            for hs in hash_size:
                card_hash = ih.phash(img_card, hash_size=hs)
                card_info['card_hash_%d' % hs] = card_hash
            new_pool.loc[0 if new_pool.empty else new_pool.index.max() + 1] = card_info

    if save_to is not None:
        new_pool.to_pickle(save_to)
    return new_pool


def fetch_all_cards_text(csv_name = "new"):
    """
    Given the query URL using Scryfall API, aggregate all card information and convert them from json to table
    :param url: query URL
    :param csv_name: path of the csv file to save the result
    :return: pandas dataframe of the fetch cards
    """
    with open('./_data/json/default-cards-20241218223248.json', 'r', encoding='utf-8') as f:
        cards = json.load(f)
        df = pd.DataFrame.from_dict(cards)
        print(f"Number of cards fetched: {len(df)}")

        # Positive Filters
        df = df[df['frame'].isin(["2003", "2015"])]
        df = df[df['border_color'].isin(["black", "borderless"])]
        df = df[df['set_type'].isin(['expansion', 'commander', 'masters', 'draft_innovation', 'core', 'masterpiece'])]
        df = df[df['lang'] == 'en']

        # Negative Filters
        df = df[df['digital'] == False]
        df = df[df['oversized'] == False]
        df = df[~df['layout'].isin(['art_series', 'token', "reversible_card"])]
        df = df[~df['name'].isin(["Forest", "Swamp", "Island", "Mountain", "Plains"])]
        print(f"Number of cards fetched for existing filters: {len(df)}")

        #print(f"Number of cards fetched for new filter: {len(df)}")

        # Keep only the useful columns
        df = df[['id', 'layout', 'card_faces', 'name', 'set', "image_uris"]]
        df.to_csv(csv_name, sep=';')  # Comma seperator doesn't work, since some columns are saved as a dict

        return df

def load_all_cards_text(csv_name):
    df = pd.read_csv(csv_name, sep=';')   # Comma seperator doesn't work, since some columns are saved as a dict
    return df


def fetch_all_cards_image(df, out_dir=None, size='png'):
    """
    Download card images from Scryfall database
    :param df: pandas dataframe (or series) of cards
    :param out_dir: path of output directory
    :param size: Image format given by Scryfall API - 'png', 'large', 'normal', 'small', 'art_crop', 'border_crop'
    :return:
    """
    if size != 'png':
        print('Note: this repo has been implemented using only \'png\' size. '
              'Using %s may result in an unexpected behaviour in other parts of this repo.' % size)
        

    if isinstance(df, pd.Series):
        # df is a single row of card
        fetch_card_image(df, out_dir, size)
    else:
        # df is a dataframe containing list of cards
        for ind, row in df.iterrows():
            fetch_card_image(row, out_dir, size)


def fetch_card_image(row, out_dir=None, size='png'):
    """
    Download a card's image from Scryfall database
    :param row: pandas series including the card's information
    :param out_dir: path of the output directory
    :param size: Image format given by Scryfall API - 'png', 'large', 'normal', 'small', 'art_crop', 'border_crop'
    :return:
    """
    out_dir = '%s/imgs/%s' % (Config.data_dir, size)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Extract card's name and URL for image accordingly
    # Double-faced cards have a different format, and results in two separate card images
    png_urls = []
    card_names = []
    if row['layout'] in ['transform', 'double_faced_token', 'modal_dfc']:
        if isinstance(row['card_faces'], str):  # For some reason, dict isn't being parsed in the previous step
            card_faces = ast.literal_eval(row['card_faces'])
        else:
            card_faces = row['card_faces']
        for i in range(len(card_faces)):
            png_urls.append(card_faces[i]['image_uris'][size])
            card_names.append('%s-%s' % (row['id'], i))
    else: #if row['layout'] == 'normal':
        try:
            if isinstance(row['image_uris'], str):  # For some reason, dict isn't being parsed in the previous step
                png_urls.append(ast.literal_eval(row['image_uris'])[size])
            else:
                png_urls.append(row['image_uris'][size])
            card_names.append((row['id']))
        except:
            print(row)

    for i in range(len(png_urls)):
        img_name = '%s/%s.png' % (out_dir, card_names[i])
        if not os.path.isfile(img_name):
            request.urlretrieve(png_urls[i], filename=img_name)
            print(img_name)

def main():
    pck_path = os.path.abspath('card_pool.pck')
    print('Warning: pickle for card database %s is not found!' % pck_path)

    #TODO: Make "new" and "old" where old is old-bordered and such.
    set_name = "new"
    csv_name = '%s/csv/%s.csv' % (Config.data_dir, set_name)
    print(csv_name)
    df = fetch_all_cards_text(csv_name=csv_name)

    fetch_all_cards_image(df, out_dir='%s/card_img/png/%s' % (Config.data_dir, set_name))

    card_pool = df
    card_pool.reset_index(drop=True, inplace=True)
    card_pool.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
    calc_image_hashes(card_pool, save_to=pck_path, hash_size=16)
    return


if __name__ == '__main__':
    main()
