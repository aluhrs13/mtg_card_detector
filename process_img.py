import numpy as np
import cv2
from ocr_helpers import four_point_transform
from PIL import Image
import imagehash as ih
import easyocr
from thefuzz import process

def id_card(img, cnt, card_pool, hash_size=16):
    pts = np.float32([p[0] for p in cnt])
    img_warp = four_point_transform(img, pts)

    crop_size = 8
    if img_warp.shape[0] > 2 * crop_size and img_warp.shape[1] > 2 * crop_size:
        img_warp = img_warp[crop_size:-crop_size, crop_size:-crop_size]
    cv2.imshow('Actual Analyzed Image', img_warp)

    img_card = Image.fromarray(img_warp.astype('uint8'), 'RGB')

    card_hash = ih.phash(img_card, hash_size=hash_size)#.hash.flatten()
    card_pool['hash_diff'] = card_pool['card_hash_%d' % hash_size]
    card_pool['hash_diff'] = card_pool['hash_diff'].apply(lambda x: x-card_hash)
    
    top_cards = card_pool.nsmallest(15, 'hash_diff')
    top_cards = top_cards[['name', 'set', 'id', 'hash_diff']].to_numpy()

    return top_cards

def process_img(img, settings):
    max_val = settings['max_val']
    kernel_size = settings['kernel_size']
    adaptive_method = settings['adaptive_method']
    block_size = settings['block_size']
    c = settings['c']
    blur = settings['blur']
    min_contour_size = settings['min_contour_size']
    adaptive_types = {0: cv2.ADAPTIVE_THRESH_MEAN_C, 1: cv2.ADAPTIVE_THRESH_GAUSSIAN_C}

    if blur % 2 == 0:
        blur += 1  # Blur size must be odd
    if block_size % 2 == 0 or block_size < 3:
        block_size = max(3, block_size + 1)  # Block size must be odd and >= 3

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    img_prev = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_prev = cv2.Canny(img_prev, 50, 150)
    img_prev = cv2.medianBlur(img_prev, blur)
    img_prev = cv2.adaptiveThreshold(
        img_prev, max_val, adaptive_method, cv2.THRESH_BINARY_INV, block_size, c)
    img_prev = cv2.dilate(img_prev, kernel, iterations=1)
    img_prev = cv2.erode(img_prev, kernel, iterations=1)

    cnts, _ = cv2.findContours(img_prev, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= min_contour_size]
    #Only Vertical cards
    cnts = [cnt for cnt in cnts if cv2.boundingRect(cnt)[2] <= cv2.boundingRect(cnt)[3]]

    img_erode_bgr = cv2.cvtColor(img_prev, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(cnts[:5]):
        x, y, w, h = cv2.boundingRect(cnt)
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.rectangle(img_erode_bgr, (x, y), (x + w, y + h), color, 5)

    height, width = img_erode_bgr.shape[:2]
    new_height = 800
    new_width = int((new_height / height) * width)
    img_resized = cv2.resize(img_erode_bgr, (new_width, new_height))

    return img_resized, cnts


def ocr_card_name(image, detail=False):
    height, width, _ = image.shape
    name_area = image[0:int(height * 0.33), 0:width]

    cv2.imshow('Name', name_area)
    reader = easyocr.Reader(['en'])
    card_text = reader.readtext(name_area, detail=0)

    if not card_text:
        return None

    card_name = ' '.join(card_text)

    return card_name

def find_closest_match(ocr_name, name_list):
    print(ocr_name)
    print(name_list)
    return process.extract(ocr_name, name_list)
