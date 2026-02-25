import sys
import cv2 as cv
import numpy as np
from pathlib import Path
from collections import Counter


def sample_circle_color(image, cx, cy, radius):
    """Return the average BGR color of pixels inside the circle."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.circle(mask, (cx, cy), max(1, radius - 2), 255, -1)
    pixels = image[mask == 255]
    if len(pixels) == 0:
        return image[cy, cx]
    return pixels.mean(axis=0).astype(np.uint8)


def classify_color(bgr):
    """Classify a BGR color into a named color category using HSV."""
    pixel = np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]])
    hsv = cv.cvtColor(pixel, cv.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 50:
        return 'black'
    if s < 30:
        return 'gray'
    # Pink: near-red hue, low saturation, bright (e.g. light pink)
    if (h > 150 or h < 15) and s < 80 and v > 190:
        return 'pink'
    # Red: near-red hue, highly saturated (s > 180 separates red from skin/peach)
    if (h < 12 or h > 160) and s > 180:
        return 'red'
    # Skin/peach/brown: warm hue, moderate saturation
    if h < 15 and s > 80:
        return 'brown'
    # Purple/violet
    if 120 <= h <= 160 and s > 60:
        return 'purple'
    # Yellow/gold (also covers cream at lower saturation)
    if 15 <= h <= 45 and s > 60:
        return 'yellow'
    if s < 80:
        return 'gray'
    return 'unknown'


def main(argv: list[str]) -> int:
    default_file = 'dots.png'
    filename = argv[0] if argv else default_file

    src = cv.imread(str(Path(filename)), cv.IMREAD_COLOR)
    if src is None:
        print(f'Error opening image: {filename}')
        return -1

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    rows = gray.shape[0]

    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, 1, rows / 128,
        param1=80, param2=30,
        minRadius=5, maxRadius=15
    )

    if circles is None:
        print('No circles detected.')
        return 0

    circles = np.uint16(np.around(circles))
    color_counts = Counter()

    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        avg_bgr = sample_circle_color(src, cx, cy, r)
        color = classify_color(avg_bgr)
        color_counts[color] += 1

    total = len(circles[0])
    print(f'Detected {total} circles:\n')
    for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
        print(f'  {color}: {count}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
