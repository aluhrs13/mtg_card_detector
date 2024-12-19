import cv2
from PIL import Image
import imagehash
import os

def main():
    video_path = 'F:\\Repos\\mtg_card_detector\\test_file\\videos\\IMG_6575.MP4'  # Replace with your video file path
    output_dir = 'output_images'      # Directory to save images

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    prev_hash = None
    last_saved_hash = None
    frame_count = 0
    card_count = 0
    threshold = 5  # Threshold for hash difference between frames
    stability_counter = 0
    stability_threshold = 5  # Number of consecutive stable frames required
    hash_diff_threshold = 10  # Threshold to detect a new card

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert frame to RGB PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Compute perceptual hash
        curr_hash = imagehash.phash(pil_image)

        if prev_hash is not None:
            # Compute hash difference with previous frame
            hash_diff = curr_hash - prev_hash

            if hash_diff < threshold:
                # Frame is similar to previous frame
                stability_counter += 1
            else:
                # Frame is different, reset stability counter
                stability_counter = 0

            if stability_counter >= stability_threshold:
                # Card is stable in frame
                if last_saved_hash is None or (curr_hash - last_saved_hash) > hash_diff_threshold:
                    # New card detected, save image
                    card_count += 1
                    image_path = os.path.join(output_dir, f'card_{card_count}.png')
                    pil_image.save(image_path)
                    print(f"Saved {image_path}")
                    last_saved_hash = curr_hash
                    # Reset stability counter
                    stability_counter = 0
                    # Optionally, skip ahead
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + stability_threshold)
                else:
                    # Same card as last saved, do not save again
                    pass
        else:
            # First frame, initialize stability counter
            stability_counter = 0

        prev_hash = curr_hash

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
