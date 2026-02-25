import sys
import cv2 as cv
import numpy as np
from pathlib import Path

def main(argv: list[str]) -> int:
    default_file = 'dots.png'
    filename = argv[0] if argv else default_file

    # Loads an image
    src = cv.imread(str(Path(filename)), cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print(f'Usage: hough_circle.py [image_name -- default {default_file}] \n')
        return -1

    # Convert to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Reduce noise to avoid false circle detection
    #gray = cv.medianBlur(gray, 5)

    cv.imshow("Detected Circles", gray)
    cv.waitKey(0)
    cv.destroyAllWindows()


    rows = gray.shape[0]
    print(rows)
    circles = cv.HoughCircles(
	gray, cv.HOUGH_GRADIENT, 1, rows / 128,
        param1=80, param2=30,
        minRadius=5, maxRadius=15
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # Circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # Circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
		    

    cv.imshow("Detected Circles", src)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
