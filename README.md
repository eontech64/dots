# dots

Detects circles in an image using OpenCV's Hough Circle Transform and counts them by color.

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy

```
pip install opencv-python numpy
```

## Usage

```
python circles.py [image_file]
```

Defaults to `dots.png` if no file is provided.

## Example output

```
Detected 1794 circles:

  black: 425
  yellow: 406
  brown: 357
  red: 309
  purple: 145
  pink: 113
  gray: 39
```
