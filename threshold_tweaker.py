import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm 
from process_img import process_img, id_card

def get_cardpool():
    pck_path = os.path.abspath('card_pool.pck')
    if os.path.isfile(pck_path):
        card_pool = pd.read_pickle(pck_path)
    else:
        print('Warning: pickle for card database %s is not found! Run fetch_data!' % pck_path)

    ch_key = 'card_hash_%d' % 16
    card_pool[ch_key] = card_pool[ch_key].apply(lambda x: x.hash.flatten())

    return card_pool[['name', 'set', 'id', ch_key]]

def load_names_dict(names_path):
    if os.path.exists(names_path):
        with open(names_path, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_names_dict(names_path, names_dict):
    with open(names_path, 'w') as f:
        json.dump(names_dict, f)

def evaluate_settings(settings, image_files, folder_path, card_pool, ground_truth):
    count_correct = 0
    total_images = len(image_files)

    for image_filename in tqdm(image_files, desc='Processing Images', leave=False):
        image_path = os.path.join(folder_path, image_filename)
        img = cv2.imread(image_path)
        if img is None:
            continue
        processed_img, cnts = process_img(img, settings)
        card_name = []
        if cnts:
            top_matches = id_card(img, cnts[0], card_pool)
            card_name = top_matches[0][0]
        true_name = ground_truth.get(image_filename, '')
        if card_name == true_name:
            count_correct += 1
    accuracy = count_correct / total_images
    return accuracy

def automatic_run(image_files, folder_path, card_pool, ground_truth):
    # Define ranges for settings to test
    max_val_options = [255] # done
    type_idx_options = [1] # Done, 1
    kernel_size_options = [1] # Done, 1
    adaptive_method_options = [1] # Done, 0 for non-canny. 1 for canny
    block_size_options = [150] # Done, 50 for non-canny
    c_options = [3] # Done, 3
    blur_options = [1] # Done, 5 for non-canny
    min_contour_size_options = [100] # Done, 100 for non-canny

    adaptive_types = {0: cv2.ADAPTIVE_THRESH_MEAN_C, 1: cv2.ADAPTIVE_THRESH_GAUSSIAN_C}

    best_accuracy = 0
    best_settings = None

    import itertools
    param_grid = list(itertools.product(
        max_val_options, type_idx_options, kernel_size_options,
        adaptive_method_options, block_size_options, c_options,
        blur_options, min_contour_size_options,
    ))

    # Wrap the parameter grid loop with tqdm
    for params in tqdm(param_grid, desc='Parameter Grid'):
        settings = {
            'max_val': params[0],
            'type_idx': params[1],
            'kernel_size': params[2],
            'adaptive_method': adaptive_types[params[3]],
            'block_size': params[4],
            'c': params[5],
            'blur': params[6],
            'min_contour_size': params[7],
            'threshold_types': threshold_types,
            'adaptive_types': adaptive_types
        }
        accuracy = evaluate_settings(settings, image_files, folder_path, card_pool, ground_truth)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_settings = settings
            print(f"New best accuracy {best_accuracy*100:.2f}%!")
        else:
            print(f"Mediocre accuracy {accuracy*100:.2f}%")

    if best_settings:
        best_settings_path = os.path.join(folder_path, 'best_settings.json')
        with open(best_settings_path, 'w') as f:
            json.dump(best_settings, f, indent=4)
        print(f"Best accuracy: {best_accuracy*100:.2f}%")
        print(f"Best settings saved to {best_settings_path}")

def main(settings, image_files, card_pool, names_dict):
    auto_increment = True
    init_settings = settings

    cv2.namedWindow('Threshold Adjustments')

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
            'adaptive_types': adaptive_types,
        }

        processed_img, cnts = process_img(img, settings)

        if image_filename not in names_dict:
            names_dict[image_filename] = ''

        top_matches = []
        if cnts:
            top_matches = id_card(img, cnts[0], card_pool)
            matched_name = top_matches[0][0]
            for match_idx, match in enumerate(top_matches):
                if match[0] == names_dict[image_filename]:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.putText(processed_img, f"{match[0]} ({match[1]}): {match[3]}", (10, (15*match_idx)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imshow('Threshold Adjustments', processed_img)

        if cnts:
            x, y, w, h = cv2.boundingRect(cnts[0])
            cropped_img = img[y:y+h, x:x+w]
            cv2.imshow('Cropped Image', cropped_img)

        key = cv2.waitKey(30)
        incremented = False
        if key == 27:  # ESC key
            break
        elif key == ord('s'):
            names_dict[image_filename] = matched_name
            print(f"Saved {image_filename}: {matched_name}")
            if matched_name == names_dict.get(image_filename, ''):
                count_correct += 1
            current_image_index += 1
            incremented = True
        elif key == 32 or auto_increment:  # Space key
            current_image_index += 1
            incremented = True

        if image_filename in names_dict and matched_name == names_dict[image_filename]:
            print(f"Correct: {image_filename}: {matched_name}")
            count_correct += 1
            if not incremented:
                current_image_index += 1
        elif image_filename in names_dict:
            print(f"Wrong: {image_filename}: {matched_name} != {names_dict[image_filename]}")
        else:
            print(f"Skipped: {image_filename}")

    cv2.destroyAllWindows()
    print(f"Accuracy: {count_correct}/{total_images}")

    save_names_dict(names_path, names_dict)

if __name__ == '__main__':
    folder_path = 'F:\\Repos\\mtg_card_detector\\test_file\\basic_using_thing'
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        raise FileNotFoundError(f"No image files found in folder: {folder_path}")

    names_path = os.path.join(folder_path, 'names.json')
    ground_truth = load_names_dict(names_path)

    card_pool = get_cardpool()

    init_settings = {
        'max_val': 255,
        'type_idx': 1,
        'kernel_size': 1,
        'adaptive_method': 1,
        'block_size': 150,
        'c': 3,
        'blur': 1,
        'min_contour_size': 100,
    }

    main(init_settings, image_files, card_pool, ground_truth)
    #automatic_run(image_files, folder_path, card_pool, ground_truth)
