import argparse
import ast
import collections
import cv2
import imagehash as ih
import numpy as np
from operator import itemgetter
import os
import pandas as pd
from PIL import Image
import time

from config import Config
from ocr_helpers import find_card, four_point_transform


def draw_card_graph(exist_cards, card_pool, f_len):
    """
    Given the history of detected cards in the current and several previous frames, draw a simple graph
    displaying the detected cards with its confidence level
    :param exist_cards: History of all detected cards in the previous (f_len) frames
    :param card_pool: pandas dataframe of all card's information
    :param f_len: length of windows (in frames) to consider for confidence level
    :return:
    """
    # Lots of constants to set the dimension of each elements
    w_card = 63  # Width of the card image displayed
    h_card = 88
    gap = 25  # Offset between each elements
    gap_sm = 10  # Small offset
    w_bar = 300  # Length of the confidence bar at 100%
    h_bar = 12
    txt_scale = 0.8
    n_cards_p_col = 4  # Number of cards displayed per one column
    w_img = gap + (w_card + gap + w_bar + gap) * 2  # Dimension of the entire graph (for 2 columns)
    h_img = 480
    img_graph = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    x_anchor = gap
    y_anchor = gap

    i = 0

    # Cards are displayed from the most confident to the least
    # Confidence level is calculated by number of frames that the card was detected in
    for key, val in sorted(exist_cards.items(), key=itemgetter(1), reverse=True)[:n_cards_p_col * 2]:
        card_name = key[:key.find('(') - 1]
        card_set = key[key.find('(') + 1:key.find(')')]
        confidence = sum(val) / f_len
        card_info = card_pool[(card_pool['name'] == card_name) & (card_pool['set'] == card_set)].iloc[0]
        img_name = '%s/imgs/tiny/%s.png' % (Config.data_dir, card_info['id'])
        # If the card image is not found, just leave it blank
        if os.path.exists(img_name):
            card_img = cv2.imread(img_name)
        else:
            card_img = np.ones((h_card, w_card, 3)) * 255
            cv2.putText(card_img, 'X', ((w_card - int(txt_scale * 25)) // 2, (h_card + int(txt_scale * 25)) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, txt_scale, (0, 0, 0), 2)

        # Insert the card image, card name, and confidence bar to the graph
        img_graph[y_anchor:y_anchor + h_card, x_anchor:x_anchor + w_card] = card_img
        cv2.putText(img_graph, '%s (%s)' % (card_name, card_set),
                    (x_anchor + w_card + gap, y_anchor + gap_sm + int(txt_scale * 25)), cv2.FONT_HERSHEY_SIMPLEX,
                    txt_scale, (255, 255, 255), 1)
        cv2.rectangle(img_graph, (x_anchor + w_card + gap, y_anchor + h_card - (gap_sm + h_bar)),
                      (x_anchor + w_card + gap + int(w_bar * confidence), y_anchor + h_card - gap_sm), (0, 255, 0),
                      thickness=cv2.FILLED)
        y_anchor += h_card + gap
        i += 1
        if i % n_cards_p_col == 0:
            x_anchor += w_card + gap + w_bar + gap
            y_anchor = gap
        pass
    return img_graph


def detect_frame(img, card_pool, hash_size=32, size_thresh=10000,
                 out_path=None, display=True, debug=True):
    """
    Identify all cards in the input frame, display or save the frame if needed
    :param img: input frame
    :param card_pool: pandas dataframe of all card's information
    :param hash_size: param for pHash algorithm
    :param size_thresh: threshold for size (in pixel) of the contour to be a candidate
    :param out_path: path to save the result
    :param display: flag for displaying the result
    :param debug: flag for debug mode
    :return: list of detected card's name/set and resulting image
    """

    img_result = img.copy()  # For displaying and saving
    det_cards = []
    # Detect contours of all cards in the image
    cnts = find_card(img_result, size_thresh=size_thresh)

    if len(cnts) == 0:
        print('No card is detected!')

    for i in range(len(cnts)):
        cnt = cnts[i]
        # For the region of the image covered by the contour, transform them into a rectangular image
        pts = np.float32([p[0] for p in cnt])
        img_warp = four_point_transform(img, pts)

        # To identify the card from the card image, perceptual hashing (pHash) algorithm is used
        # Perceptual hash is a hash string built from features of the input medium. If two media are similar
        # (ie. has similar features), their resulting pHash value will be very close.
        # Using this property, the matching card for the given card image can be found by comparing pHash of
        # all cards in the database, then finding the card that results in the minimal difference in pHash value.
        
        img_card = Image.fromarray(img_warp.astype('uint8'), 'RGB')

        # the stored values of hashes in the dataframe is pre-emptively flattened already to minimize computation time
        card_hash = ih.phash(img_card, hash_size=hash_size).hash.flatten()
        card_pool['hash_diff'] = card_pool['card_hash_%d' % hash_size]
        card_pool['hash_diff'] = card_pool['hash_diff'].apply(lambda x: np.count_nonzero(x != card_hash))
        min_card = card_pool[card_pool['hash_diff'] == min(card_pool['hash_diff'])].iloc[0]
        card_name = min_card['name']
        card_set = min_card['set']
        det_cards.append((card_name, card_set))
        hash_diff = min_card['hash_diff']
        print(card_name)

        # Render the result, and display them if needed
        cv2.drawContours(img_result, [cnt], -1, (0, 255, 0), 2)
        
        # Ensure pts contains integer coordinates
        pts = [(int(x), int(y)) for x, y in pts]
        cv2.putText(img_result, card_name, (min(pts[0][0], pts[1][0]), min(pts[0][1], pts[1][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if debug:
            cv2.putText(img_warp, card_name + ', ' + str(hash_diff), (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow('detect_frame card#%d' % i, img_warp)
    if display:
        height, width = img_result.shape[:2]
        new_height = 800
        new_width = int((new_height / height) * width)
        img_result = cv2.resize(img_result, (new_width, new_height))
        cv2.imshow('Result', img_result)
        cv2.waitKey(0)

    if out_path is not None:
        cv2.imwrite(out_path, img_result.astype(np.uint8))
    return det_cards, img_result


def detect_video(capture, card_pool, hash_size=32, size_thresh=10000,
                 out_path=None, display=True, show_graph=True, debug=False):
    """
    Identify all cards in the continuous video stream, display or save the result if needed
    :param capture: input video stream
    :param card_pool: pandas dataframe of all card's information
    :param hash_size: param for pHash algorithm
    :param size_thresh: threshold for size (in pixel) of the contour to be a candidate
    :param out_path: path to save the result
    :param display: flag for displaying the result
    :param show_graph: flag to show graph
    :param debug: flag for debug mode
    :return: list of detected card's name/set and resulting image
    :return:
    """
    # Get the dimension of the output video, and set it up
    if show_graph:
        img_graph = draw_card_graph({}, pd.DataFrame(), -1)  # Black image of the graph just to get the dimension
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) + img_graph.shape[1]
        height = max(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), img_graph.shape[0])
    else:
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if out_path is not None:
        vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (width, height))
    max_num_obj = 0
    f_len = 10  # number of frames to consider to check for existing cards
    exist_cards = {}
    try:
        while True:
            ret, frame = capture.read()
            start_time = time.time()
            if not ret:
                # End of video
                print("End of video. Press any key to exit")
                cv2.waitKey(0)
                break
            # Detect all cards from the current frame
            det_cards, img_result = detect_frame(frame, card_pool, hash_size=hash_size, size_thresh=size_thresh,
                                                 out_path=None, display=False, debug=debug)
            if show_graph:
                # If the card was already detected in the previous frame, append 1 to the list
                # If the card previously detected was not found in this trame, append 0 to the list
                # If the card wasn't previously detected, make a new list and add 1 to it
                # If the same card is detected multiple times in the same frame, keep track of the duplicates
                # The confidence will be calculated based on the number of frames the card was detected for
                det_cards_count = collections.Counter(det_cards).items()
                det_cards_list = []
                for card, count in det_cards_count:
                    card_name, card_set = card
                    for i in range(count): 1
                    key = '%s (%s) #%d' % (card_name, card_set, i + 1)
                    det_cards_list.append(key)
                gone = []
                for key, val in exist_cards.items():
                    if key in det_cards_list:
                        exist_cards[key] = exist_cards[key][1 - f_len:] + [1]
                    else:
                        exist_cards[key] = exist_cards[key][1 - f_len:] + [0]
                    if len(val) == f_len and sum(val) == 0:
                        gone.append(key)
                for key in det_cards_list:
                    if key not in exist_cards.keys():
                        exist_cards[key] = [1]
                for key in gone:
                    exist_cards.pop(key)

                # Draw the graph based on the history of detected cards, then concatenate it with the result image
                img_graph = draw_card_graph(exist_cards, card_pool, f_len)
                img_save = np.zeros((height, width, 3), dtype=np.uint8)
                img_save[0:img_result.shape[0], 0:img_result.shape[1]] = img_result
                img_save[0:img_graph.shape[0], img_result.shape[1]:img_result.shape[1] + img_graph.shape[1]] = img_graph
            else:
                img_save = img_result

            # Display the result
            if display:
                cv2.imshow('result', img_save)
            if debug:
                max_num_obj = max(max_num_obj, len(det_cards))
                for i in range(len(det_cards), max_num_obj):
                    cv2.imshow('card#%d' % i, np.zeros((1, 1), dtype=np.uint8))

            elapsed_ms = (time.time() - start_time) * 1000
            print('Elapsed time: %.2f ms' % elapsed_ms)
            if out_path is not None:
                vid_writer.write(img_save.astype(np.uint8))
            cv2.waitKey(1)
    except KeyboardInterrupt:
        capture.release()
        if out_path is not None:
            vid_writer.release()
        cv2.destroyAllWindows()


def main(args):
    # Specify paths for all necessary files

    pck_path = os.path.abspath('card_pool.pck')
    if os.path.isfile(pck_path):
        card_pool = pd.read_pickle(pck_path)
    else:
        print('Warning: pickle for card database %s is not found! Run fetch_data!' % pck_path)

    ch_key = 'card_hash_%d' % args.hash_size
    card_pool = card_pool[['name', 'set', 'id', ch_key]]

    # Processing time is almost linear to the size of the database
    # Program can be much faster if the search scope for the card can be reduced
    card_pool = card_pool[card_pool['set'].isin(Config.all_set_list)]

    # ImageHash is basically just one numpy.ndarray with (hash_size)^2 number of bits. pre-emptively flattening it
    # significantly increases speed for subtracting hashes in the future.
    card_pool[ch_key] = card_pool[ch_key].apply(lambda x: x.hash.flatten())

    # If the test file isn't given, use webcam to capture video
    if args.in_path is None:
        capture = cv2.VideoCapture(0)
        detect_video(capture, card_pool, hash_size=args.hash_size, out_path='%s/result.avi' % args.out_path,
                     display=args.display, show_graph=args.show_graph, debug=args.debug)
        capture.release()
    else:
        # Save the detection result if args.out_path is provided
        if args.out_path is None:
            out_path = None
        else:
            #TODO: Handle both image and video
            f_name = os.path.split(args.in_path)[1]
            out_path = '%s/%s.jpg' % (args.out_path, f_name[:f_name.find('.')])

        if not os.path.isfile(args.in_path):
            print('The test file %s doesn\'t exist!' % os.path.abspath(args.in_path))
            return
        # Check if test file is image or video
        test_ext = args.in_path[args.in_path.find('.') + 1:]
        if test_ext in ['jpg', 'jpeg', 'bmp', 'png', 'tiff']:
            # Test file is an image
            img = cv2.imread(args.in_path)
            detect_frame(img, card_pool, hash_size=args.hash_size, out_path=out_path, display=args.display,
                         debug=args.debug)
        else:
            # Test file is a video
            capture = cv2.VideoCapture(args.in_path)
            detect_video(capture, card_pool, hash_size=args.hash_size, out_path=out_path, display=args.display,
                         show_graph=args.show_graph, debug=args.debug)
            capture.release()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in', dest='in_path', help='Path of the input file. For webcam, leave it blank',
                        type=str, default="C:\\Users\\aluhrs\\Downloads\\ocr_test\\mtg_card_detector\\test_file\\test1.mp4")
    parser.add_argument('-o', '--out', dest='out_path', help='Path of the output directory to save the result',
                        type=str, default="_data\\output")
    parser.add_argument('-hs', '--hash_size', dest='hash_size',
                        help='Size of the hash for pHash algorithm', type=int, default=16)
    parser.add_argument('-dsp', '--display', dest='display', help='Display the result', action='store_true',
                        default=True)
    parser.add_argument('-dbg', '--debug', dest='debug', help='Enable debug mode', action='store_true', default=True)
    parser.add_argument('-gph', '--show_graph', dest='show_graph', help='Display the graph for video output', 
                        action='store_true', default=True)
    args = parser.parse_args()
    #if not args.display and args.out_path is None:
        # Then why the heck are you running this thing in the first place?
    #    print('The program isn\'t displaying nor saving any output file. Please change the setting and try again.')
    #    exit()
    main(args)
