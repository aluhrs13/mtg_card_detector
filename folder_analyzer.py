import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm 
from process_img import process_img, id_card, ocr_card_name, find_closest_match

def get_cardpool(hash_size=16):
    pck_path = os.path.abspath(f'.\_data\pickles\card_pool_{hash_size}.pck')
    if os.path.isfile(pck_path):
        card_pool = pd.read_pickle(pck_path)
    else:
        print('Warning: pickle for card database %s is not found! Run fetch_data!' % pck_path)

    print('Loaded card pool with %d cards' % len(card_pool))
    ch_key = 'card_hash_%d' % hash_size
    return card_pool[['name', 'set', 'id', ch_key]]

def add_name(folder_path, img_name, card_name):
    names_file = os.path.join(folder_path, 'names.json')
    if os.path.isfile(names_file):
        with open(names_file, 'r') as f:
            names_data = json.load(f)
    else:
        names_data = []

    names_data.append((img_name, card_name))

    with open(names_file, 'w') as f:
        json.dump(names_data, f, indent=4)

def main(image_files, card_pool, names_data):
    auto_increment = False
    debug = True

    if debug:
        cv2.namedWindow('Threshold Adjustments')
        cv2.createTrackbar('Kernel Size', 'Threshold Adjustments', 1, 20, lambda x: None)
        cv2.createTrackbar('Block Size', 'Threshold Adjustments', 150, 50, lambda x: None)
        cv2.createTrackbar('C', 'Threshold Adjustments', 5, 20, lambda x: None)
        cv2.createTrackbar('Blur', 'Threshold Adjustments', 1, 20, lambda x: None)
        cv2.createTrackbar('Min Contour Size', 'Threshold Adjustments', 100, 1000, lambda x: None)

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

        if debug:
            kernel_size = cv2.getTrackbarPos('Kernel Size', 'Threshold Adjustments') # -1 to 30
            block_size = cv2.getTrackbarPos('Block Size', 'Threshold Adjustments') # -1 to 300
            c = cv2.getTrackbarPos('C', 'Threshold Adjustments')# -1 to 20
            blur = cv2.getTrackbarPos('Blur', 'Threshold Adjustments')# -1 to 20
            min_contour_size = cv2.getTrackbarPos('Min Contour Size', 'Threshold Adjustments')
        else:
            kernel_size = -1
            block_size = 50
            c = 5
            blur = -1
            min_contour_size = 100
            canny = False

        processed_img, cnts = process_img(img, kernel_size=kernel_size, block_size=block_size, c=c, blur=blur, min_contour_size=min_contour_size, debug=debug, canny=canny)

        ocr_name = ""
        matched_name = "<SKIP>"
        top_matches = []
        if cnts:
            # Crop to top corner for OCR
            x, y, w, h = cv2.boundingRect(cnts[0])
            cropped_img = img[y:y+h, x:x+w]
            ocr_text = ocr_card_name(cropped_img, debug=debug)

            # Get Phash match
            top_matches = id_card(img, cnts[0], card_pool, debug=debug, crop_size=8)

            if ocr_text is not None:
                ocr_matches = find_closest_match(ocr_text, [match[0] for match in top_matches])

                if ocr_matches[0][1] > 75:
                    ocr_name = ocr_matches[0][0]
                    matched_name = ocr_name

            for match_idx, match in enumerate(top_matches):
                overlay_str = f"{match[0]} ({match[1]}): {match[3]}"

                if match[0] == ocr_name:
                    overlay_str = "(OCR)" + overlay_str
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                if debug:
                    cv2.putText(processed_img, overlay_str, (10, (15*match_idx)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if debug:
            cv2.imshow('Threshold Adjustments', processed_img)

        if auto_increment:
            if matched_name == names_data[image_files[current_image_index]]:
                count_correct += 1
            else:
                print(f"{image_files[current_image_index]}({names_data[image_files[current_image_index]]}) - {matched_name}")
            current_image_index += 1
        else:
            key = cv2.waitKey(30)
            if key == 27:  # ESC key
                break
            elif key == ord('a'):
                #add_name(folder_path, image_filename, "")
                current_image_index += 1
            elif key == ord('d'):
                #add_name(folder_path, image_filename, matched_name)
                print("Saving name: ", matched_name)
                current_image_index += 1

    cv2.destroyAllWindows()
    print(f"Accuracy: {count_correct}/{total_images}")

if __name__ == '__main__':
    folder_path = 'F:\\Repos\\mtg_card_detector\\_data\\output_images'
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        raise FileNotFoundError(f"No image files found in folder: {folder_path}")

    card_pool = get_cardpool()

    name_file = os.path.join(folder_path, 'names.json')
    if os.path.isfile(name_file):
        with open(name_file, 'r') as f:
            names_data = json.load(f)

    main(image_files, card_pool, names_data)
