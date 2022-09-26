# Barcodes and QR-Codes
### by Louis Heredero

This repository gathers all files related to my maturity work.

The main PDF file is [latex/main.pdf](latex/main.pdf)

## LaTeX
All latex files, including the [bibliographic database](latex/db.bib), are located in the [latex](latex) folder. The various figures and images are placed in [latex/images](latex/images).

## Python
Some LaTeX tables have been created using Python, as well as most of the figures.
The Python scripts used for this purpose are in the folders [python/latex_gen](python/latex_gen) and [python/img_gen](python/img_gen), respectively.

The other scripts used for code generation or scanning are simply put in the [python](python) directory.

### Dependencies

I used Python 3.6 to run the scripts but any Python version above 3 should work.

Most scripts require [Pygame](https://pypi.org/project/pygame/) and [NumPy](https://pypi.org/project/numpy/).
Some also need [imutils](https://pypi.org/project/imutils/).

The scanners work using [OpenCV](https://pypi.org/project/opencv-python/)

### Scanners

The QR-Code scanner can currently only read version 1 codes. This will be improved in the near future.

The Lycacode scanners, both normal and mini, require that the code is as flat as possible. It may not work properly if there are bumps, folds or curvature on the code.

## Copyright
All rights are reserved to the author. For inquiries, please contact me through the [discussions tab](https://github.com/LordBaryhobal/5D_Heredero_Louis_TM2022/discussions) or at lordbaryhobal+tm@gmail.com
