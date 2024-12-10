import os
import json
import cv2
import numpy as np
import imagehash as ih
import pandas as pd
from PIL import Image
from ocr_helpers import four_point_transform

def get_cardpool():
    pck_path = os.path.abspath('card_pool.pck')
    if os.path.isfile(pck_path):
        card_pool = pd.read_pickle(pck_path)
    else:
        print('Warning: pickle for card database %s is not found! Run fetch_data!' % pck_path)

    ch_key = 'card_hash_%d' % 16
    card_pool[ch_key] = card_pool[ch_key].apply(lambda x: x.hash.flatten())

    return card_pool[['name', 'set', 'id', ch_key]]

def id_card(img, cnt, card_pool):
    pts = np.float32([p[0] for p in cnt])
    img_warp = four_point_transform(img, pts)

    img_card = Image.fromarray(img_warp.astype('uint8'), 'RGB')

    card_hash = ih.phash(img_card, hash_size=16).hash.flatten()
    card_pool['hash_diff'] = card_pool['card_hash_%d' % 16]
    card_pool['hash_diff'] = card_pool['hash_diff'].apply(lambda x: np.count_nonzero(x != card_hash))
    
    top_cards = card_pool.nsmallest(5, 'hash_diff')
    min_card = top_cards.iloc[0]
    card_name = min_card['name']
    card_set = min_card['set']
    hash_diff = min_card['hash_diff']

    return card_name

def process_image(img, settings):
    max_val = settings['max_val']
    type_idx = settings['type_idx']
    kernel_size = settings['kernel_size']
    adaptive_method = settings['adaptive_method']
    block_size = settings['block_size']
    c = settings['c']
    blur = settings['blur']
    min_contour_size = settings['min_contour_size']
    threshold_types = settings['threshold_types']
    adaptive_types = settings['adaptive_types']

    if blur % 2 == 0:
        blur += 1  # Blur size must be odd
    if block_size % 2 == 0 or block_size < 3:
        block_size = max(3, block_size + 1)  # Block size must be odd and >= 3

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, blur)
    img_thresh = cv2.adaptiveThreshold(
        img_blur, max_val, adaptive_method, threshold_types[type_idx], block_size, c)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    cnts, _ = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= min_contour_size]

    img_erode_bgr = cv2.cvtColor(img_erode, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(cnts[:5]):
        x, y, w, h = cv2.boundingRect(cnt)
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.rectangle(img_erode_bgr, (x, y), (x + w, y + h), color, 5)

    height, width = img_erode_bgr.shape[:2]
    new_height = 800
    new_width = int((new_height / height) * width)
    img_resized = cv2.resize(img_erode_bgr, (new_width, new_height))

    return img_resized, cnts

def load_names_dict(names_path):
    if os.path.exists(names_path):
        with open(names_path, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_names_dict(names_path, names_dict):
    with open(names_path, 'w') as f:
        json.dump(names_dict, f)

def main():
    auto_increment = True
    folder_path = 'F:\\Repos\\mtg_card_detector\\test_file\\basic'
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        raise FileNotFoundError(f"No image files found in folder: {folder_path}")

    names_path = os.path.join(folder_path, 'names.json')
    names_dict = load_names_dict(names_path)

    card_pool = get_cardpool()

    cv2.namedWindow('Threshold Adjustments')

    init_settings = {
        'max_val': 255,
        'type_idx': 1,
        'kernel_size': 3,
        'adaptive_method': 0,
        'block_size': 15,
        'c': 2,
        'blur': 5,
        'min_contour_size': 200,
    }

    threshold_types = {0: cv2.THRESH_BINARY, 1: cv2.THRESH_BINARY_INV}
    adaptive_types = {0: cv2.ADAPTIVE_THRESH_MEAN_C, 1: cv2.ADAPTIVE_THRESH_GAUSSIAN_C}

    cv2.createTrackbar('Max Value', 'Threshold Adjustments', init_settings['max_val'], 255, lambda x: None)
    cv2.createTrackbar('Type', 'Threshold Adjustments', init_settings['type_idx'], len(threshold_types)-1, lambda x: None)
    cv2.createTrackbar('Kernel Size', 'Threshold Adjustments', init_settings['kernel_size'], 20, lambda x: None)
    cv2.createTrackbar('Adaptive Method', 'Threshold Adjustments', init_settings['adaptive_method'], len(adaptive_types)-1, lambda x: None)
    cv2.createTrackbar('Block Size', 'Threshold Adjustments', init_settings['block_size'], 50, lambda x: None)
    cv2.createTrackbar('C', 'Threshold Adjustments', init_settings['c'], 20, lambda x: None)
    cv2.createTrackbar('Blur', 'Threshold Adjustments', init_settings['blur'], 20, lambda x: None)
    cv2.createTrackbar('Min Contour Size', 'Threshold Adjustments', init_settings['min_contour_size'], 1000, lambda x: None)

    current_image_index = 0
    count_correct = 0
    total_images = len(image_files)

    while current_image_index < total_images:
        image_filename = image_files[current_image_index]
        image_path = os.path.join(folder_path, image_filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            current_image_index += 1
            continue

        settings = {
            'max_val': cv2.getTrackbarPos('Max Value', 'Threshold Adjustments'),
            'type_idx': cv2.getTrackbarPos('Type', 'Threshold Adjustments'),
            'kernel_size': cv2.getTrackbarPos('Kernel Size', 'Threshold Adjustments'),
            'adaptive_method': adaptive_types[cv2.getTrackbarPos('Adaptive Method', 'Threshold Adjustments')],
            'block_size': cv2.getTrackbarPos('Block Size', 'Threshold Adjustments'),
            'c': cv2.getTrackbarPos('C', 'Threshold Adjustments'),
            'blur': cv2.getTrackbarPos('Blur', 'Threshold Adjustments'),
            'min_contour_size': cv2.getTrackbarPos('Min Contour Size', 'Threshold Adjustments'),
            'threshold_types': threshold_types,
            'adaptive_types': adaptive_types
        }

        processed_img, cnts = process_image(img, settings)

        card_name = ''
        if cnts:
            card_name = id_card(img, cnts[0], card_pool)
            cv2.putText(processed_img, card_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Threshold Adjustments', processed_img)

        key = cv2.waitKey(30)
        if key == 27:  # ESC key
            break
        elif key == ord('s'):
            names_dict[image_filename] = card_name
            print(f"Saved {image_filename}: {card_name}")
            if card_name == names_dict.get(image_filename, ''):
                count_correct += 1
            current_image_index += 1
        elif key == 32 or auto_increment:  # Space key
            current_image_index += 1

        if image_filename in names_dict and card_name == names_dict[image_filename]:
            print(f"Correct: {image_filename}: {card_name}")
            count_correct += 1
        elif image_filename in names_dict:
            print(f"Wrong: {image_filename}: {card_name} != {names_dict[image_filename]}")
        else:
            print(f"Skipped: {image_filename}")

    cv2.destroyAllWindows()
    print(f"Accuracy: {count_correct}/{total_images}")

    #save_names_dict(names_path, names_dict)

if __name__ == '__main__':
    main()
