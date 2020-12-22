# OmniPrint 



README not ready yet...


A synthetic data generator for text recognition



## Installation

- Clone this repo:
```zsh
git clone https://github.com/SunHaozhe/OmniPrint
cd OmniPrint
pip install -r requirements.txt
```


## Getting Started

Edit `quick_run.py` to set up different options, then run:

```zsh
python3 quick_run.py  
```

Or specify the options directly using the command line interface: 

```zsh
python3 run.py --count 10
```

```
usage: run.py [-h] [--output_dir [OUTPUT_DIR]] [-i [INPUT_FILE]]
              [-l [LANGUAGE]] -c [COUNT] [-rs] [-let] [-num] [-sym]
              [-w [LENGTH]] [-r] [-s [SIZE]] [-p [NB_PROCESSES]]
              [-e [EXTENSION]] [-wk] [-bl [BLUR]] [-rbl] [-b [BACKGROUND]]
              [-hw] [-om] [-d [DISTORSION]] [-do [DISTORSION_ORIENTATION]]
              [-m [MARGINS]] [-ft [FONT]] [-fd [FONT_DIR]]
              [-fidx [FONT_INDEX]] [-id [IMAGE_DIR]] [-ca [CASE]] [-dt [DICT]]
              [-fwt [FONT_WEIGHT]] [-stf [STROKE_FILL]] [-im [IMAGE_MODE]]
              [-rsd RANDOM_SEED] [-esl] [-otlwd OUTLINE_WIDTH]
              [-fsz FONT_SIZE] [-lt [LINEAR_TRANSFORM]] [-rtn [ROTATION]]
              [-rrtn] [-shrx [SHEAR_X]] [-rshrx] [-shry [SHEAR_Y]] [-rshry]
              [-sclx [SCALE_X]] [-rsclx] [-scly [SCALE_Y]] [-rscly]
              [-alpha [ALPHA]] [-ralpha] [-beta [BETA]] [-rbeta]
              [-gamma [GAMMA]] [-rgamma] [-delta [DELTA]] [-rdelta] [-rtslnx]
              [-rtslny] [-pt [PERSPECTIVE_TRANSFORM]]
              [-rpt [RANDOM_PERSPECTIVE_TRANSFORM]]
              [-gpr GAUSSIAN_PRIOR_RESIZING] [-morphero MORPH_EROSION]
              [-rmorphero] [-morphdil MORPH_DILATION] [-rmorphdil]
              [-morphope MORHP_OPENING] [-rmorphope] [-morphclo MORHP_CLOSING]
              [-rmorphclo] [-morphgra MORHP_GRADIENT] [-rmorphgra]
              [-morphtoph MORHP_TOPHAT] [-rmorphtoph]
              [-morphblah MORHP_BLACKHAT] [-rmorphblah]

Generate synthetic text data for text recognition.

optional arguments:
  -h, --help            show this help message and exit
  --output_dir [OUTPUT_DIR]
                        The output directory
  -i [INPUT_FILE], --input_file [INPUT_FILE]
                        When set, this argument uses a specified text file as
                        source for the text
  -l [LANGUAGE], --language [LANGUAGE]
                        The language to use, should be fr (French), en
                        (English), es (Spanish), de (German), ar (Arabic), cn
                        (Chinese), or hi (Hindi)
  -c [COUNT], --count [COUNT]
                        The number of images to be created.
  -rs, --random_sequences
                        Use random sequences as the source text for the
                        generation. Set '-let','-num','-sym' to use
                        letters/numbers/symbols. If none specified, using all
                        three.
  -let, --include_letters
                        Define if random sequences should contain letters.
                        Only works with -rs
  -num, --include_numbers
                        Define if random sequences should contain numbers.
                        Only works with -rs
  -sym, --include_symbols
                        Define if random sequences should contain symbols.
                        Only works with -rs
  -w [LENGTH], --length [LENGTH]
                        Define how many words should be included in each
                        generated sample. If the text source is Wikipedia,
                        this is the MINIMUM length
  -r, --random          Define if the produced string will have variable word
                        count (with --length being the maximum)
  -s [SIZE], --size [SIZE]
                        Define the height of the produced images if
                        horizontal, else the width
  -p [NB_PROCESSES], --nb_processes [NB_PROCESSES]
                        Define the number of processes to use for image
                        generation. If not provided, this equals to the number
                        of CPU cores
  -e [EXTENSION], --extension [EXTENSION]
                        Define the extension to save the image with
  -wk, --use_wikipedia  Use Wikipedia as the source text for the generation,
                        using this paremeter ignores -r, -n, -s
  -bl [BLUR], --blur [BLUR]
                        Apply gaussian blur to the resulting sample. Should be
                        an integer defining the blur radius, 0 by default.
  -rbl, --random_blur   When set, the blur radius will be randomized between 0
                        and -bl.
  -b [BACKGROUND], --background [BACKGROUND]
                        Define what kind of background to use. 0: Gaussian
                        Noise, 1: Plain white, 2: Quasicrystal, 3: Image
  -hw, --handwritten    Define if the data will be "handwritten" by an RNN
  -om, --output_mask    Define if the generator will return masks for the text
  -d [DISTORSION], --distorsion [DISTORSION]
                        Define a distorsion applied to the resulting image. 0:
                        None (Default), 1: Sine wave, 2: Cosine wave, 3:
                        Random
  -do [DISTORSION_ORIENTATION], --distorsion_orientation [DISTORSION_ORIENTATION]
                        Define the distorsion's orientation. Only used if -d
                        is specified. 0: Vertical (Up and down), 1: Horizontal
                        (Left and Right), 2: Both
  -m [MARGINS], --margins [MARGINS]
                        Define the margins (percentage) around the text when
                        rendered. Each element should be a float
  -ft [FONT], --font [FONT]
                        Define font to be used
  -fd [FONT_DIR], --font_dir [FONT_DIR]
                        Define a font directory to be used
  -fidx [FONT_INDEX], --font_index [FONT_INDEX]
                        Define the font index file to be used, an example is
                        fonts/latin.txt
  -id [IMAGE_DIR], --image_dir [IMAGE_DIR]
                        Define an image directory to use when background is
                        set to image
  -ca [CASE], --case [CASE]
                        Generate upper or lowercase only. arguments: upper or
                        lower. Example: --case upper
  -dt [DICT], --dict [DICT]
                        Define the dictionary to be used
  -fwt [FONT_WEIGHT], --font_weight [FONT_WEIGHT]
                        Define the width of the strokes
  -stf [STROKE_FILL], --stroke_fill [STROKE_FILL]
                        Define the color of the strokes
  -im [IMAGE_MODE], --image_mode [IMAGE_MODE]
                        Define the image mode to be used. RGB is default, L
                        means 8-bit grayscale images, 1 means 1-bit binary
                        images stored with one pixel per byte, etc.
  -rsd RANDOM_SEED, --random_seed RANDOM_SEED
                        Random seed
  -esl, --ensure_square_layout
                        Whether the width should be the same as the height
  -otlwd OUTLINE_WIDTH, --outline_width OUTLINE_WIDTH
                        Width of stroke outline. Not yet implemented
  -fsz FONT_SIZE, --font_size FONT_SIZE
                        Font size in point
  -lt [LINEAR_TRANSFORM], --linear_transform [LINEAR_TRANSFORM]
                        The parameter for linear transform. The length must be
                        4, 5 or 9. Length 4 corresponds low level
                        parameterization, which means a, b, d, e, this is the
                        most customizable parameterization. Length 5 and
                        length 9 correspond to high level parameterization.
                        Length 5 means rotation, shear_x, shear_y, scale_x,
                        scale_y. Length 9 means rotation, shear_x, shear_y,
                        scale_x, scale_y, alpha_, beta_, gamma_, delta_. If
                        this parameter is set, other linear transform
                        parameters like rotation, shear_x, etc. will be
                        ignored
  -rtn [ROTATION], --rotation [ROTATION]
                        Define rotation angle (in degree) of the generated
                        text. Used only when --linear_transform is not set
  -rrtn, --random_rotation
                        Uniformly sample the value of rotation, the parameter
                        --rotation needs to be set. The range is
                        [-abs(rotation), abs(rotation)]
  -shrx [SHEAR_X], --shear_x [SHEAR_X]
                        High level linear transform parameter, horizontal
                        shear. Used only when --linear_transform is not set
  -rshrx, --random_shear_x
                        Uniformly sample the value of shear_x, the parameter
                        --shear_x needs to be set. The range is
                        [-abs(shear_x), abs(shear_x)]
  -shry [SHEAR_Y], --shear_y [SHEAR_Y]
                        High level linear transform parameter, vertical shear.
                        Used only when --linear_transform is not set
  -rshry, --random_shear_y
                        Uniformly sample the value of shear_y, the parameter
                        --shear_y needs to be set. The range is
                        [-abs(shear_y), abs(shear_y)]
  -sclx [SCALE_X], --scale_x [SCALE_X]
                        High level linear transform parameter, horizontal
                        scale. Used only when --linear_transform is not set
  -rsclx, --random_scale_x
                        Uniformly sample the value of scale_x, the parameter
                        --scale_x needs to be set. The range is
                        [-abs(scale_x), abs(scale_x)]
  -scly [SCALE_Y], --scale_y [SCALE_Y]
                        High level linear transform parameter, vertical scale.
                        Used only when --linear_transform is not set
  -rscly, --random_scale_y
                        Uniformly sample the value of scale_y, the parameter
                        --scale_y needs to be set. The range is
                        [-abs(scale_y), abs(scale_y)]
  -alpha [ALPHA], --alpha [ALPHA]
                        Customizable high level linear transform parameter,
                        top left element in the 2x2 matrix. Used only when
                        --linear_transform is not set
  -ralpha, --random_alpha
                        Uniformly sample the value of alpha, the parameter
                        --alpha needs to be set. The range is [-abs(alpha),
                        abs(alpha)]
  -beta [BETA], --beta [BETA]
                        Customizable high level linear transform parameter,
                        top right element in the 2x2 matrix. Used only when
                        --linear_transform is not set
  -rbeta, --random_beta
                        Uniformly sample the value of beta, the parameter
                        --beta needs to be set. The range is [-abs(beta),
                        abs(beta)]
  -gamma [GAMMA], --gamma [GAMMA]
                        Customizable high level linear transform parameter,
                        bottom left element in the 2x2 matrix. Used only when
                        --linear_transform is not set
  -rgamma, --random_gamma
                        Uniformly sample the value of gamma, the parameter
                        --gamma needs to be set. The range is [-abs(gamma),
                        abs(gamma)]
  -delta [DELTA], --delta [DELTA]
                        Customizable high level linear transform parameter,
                        bottom right element in the 2x2 matrix. Used only when
                        --linear_transform is not set
  -rdelta, --random_delta
                        Uniformly sample the value of delta, the parameter
                        --delta needs to be set. The range is [-abs(delta),
                        abs(delta)]
  -rtslnx, --random_translation_x
                        Uniformly sample the value of horizontal translation.
                        This will have no effect if horizontal margins are 0
  -rtslny, --random_translation_y
                        Uniformly sample the value of vertical translation.
                        This will have no effect if vertical margins are 0
  -pt [PERSPECTIVE_TRANSFORM], --perspective_transform [PERSPECTIVE_TRANSFORM]
                        Given the coordinates of the four corners of the first
                        quadrilateral and the coordinates of the four corners
                        of the second quadrilateral, perform the perspective
                        transform that maps a new point in the first
                        quadrilateral onto the appropriate position on the
                        second quadrilateral. Perspective transformation
                        simulates different angle of view. Enter 8 real
                        numbers (float) which correspond to the 4 corner
                        points (2D coordinates) of the target quadrilateral,
                        these 4 corner points which be respectively mapped to
                        [[0, 0], [1, 0], [0, 1], [1, 1]] in the source
                        quadrilateral. [0, 0] is the top left corner, [1, 0]
                        is the top left corner, [0, 1] is the bottom left
                        corner, [1, 1] is the bottom right corner.These
                        coordinates have been normalized to unit square [0,
                        1]^2. Thus, the entered corner points should match the
                        order of magnitude and must be convex. For example,
                        0,0,1,0,0,1,1,1 will produce identity transform. This
                        option will have no effect if
                        --random_perspective_transform is set. This option
                        sometimes will cut off the periphery of the text,
                        causing noisy data.
  -rpt [RANDOM_PERSPECTIVE_TRANSFORM], --random_perspective_transform [RANDOM_PERSPECTIVE_TRANSFORM]
                        Randomly generate a convex quadrilateral which will be
                        mapped to the normalized unit square, the value of
                        each axis is independently sampled from the gaussian
                        distribution, the standard deviation of the gaussian
                        distribution is given by
                        --random_perspective_transform. If this option is
                        present but not followed by a command-line argument,
                        the standard deviation 0.05 will be used by default.
  -gpr GAUSSIAN_PRIOR_RESIZING, --gaussian_prior_resizing GAUSSIAN_PRIOR_RESIZING
                        If not None, apply Gaussian filter to smooth image
                        prior to resizing, the argument of this parameter
                        needs to be a float, which will be used as the
                        standard deviation of Gaussian filter. Default is
                        None, which means Gaussian filter is not used before
                        resizing.
  -morphero MORPH_EROSION, --morph_erosion MORPH_EROSION
                        Morphological image processing - erosion. The argument
                        must be a tuple separated by comma without space, the
                        first element is the kernel size, the second element
                        is the number of iterations. For example, 3,2 means
                        kernel_size=3x3, iterations=2. 3,2,ellipse (3,2,cross)
                        means using elliptical (cross-shaped) kernel
                        respectively. If the third argument is not given, the
                        default kernel shape will be rectangle.
  -rmorphero, --random_morph_erosion
                        Uniformly sample the value of morphological erosion,
                        the parameter --morph_erosion needs to be set. The
                        range is [1, kernel_size] ([1, iterations])
                        kernel_shape is randomly chosen among [rectangle,
                        ellipse, cross].
  -morphdil MORPH_DILATION, --morph_dilation MORPH_DILATION
                        Morphological image processing - dilation. The
                        argument must be a tuple separated by comma without
                        space, the first element is the kernel size, the
                        second element is the number of iterations. For
                        example, 3,2 means kernel_size=3x3, iterations=2.
                        3,2,ellipse (3,2,cross) means using elliptical (cross-
                        shaped) kernel respectively. If the third argument is
                        not given, the default kernel shape will be rectangle.
  -rmorphdil, --random_morph_dilation
                        Uniformly sample the value of morphological dilation,
                        the parameter --morph_dilation needs to be set. The
                        range is [1, kernel_size] ([1, iterations])
                        kernel_shape is randomly chosen among [rectangle,
                        ellipse, cross].
  -morphope MORHP_OPENING, --morhp_opening MORHP_OPENING
                        Morphological image processing - opening. The argument
                        must be a tuple separated by comma without space, the
                        first element is the kernel size, the second element
                        is the kernel shape. For example, 3 means
                        kernel_size=3x3. 3,ellipse (3,cross) means using
                        elliptical (cross-shaped) kernel respectively. If the
                        second argument is not given, the default kernel shape
                        will be rectangle.
  -rmorphope, --random_morph_opening
                        Uniformly sample the value of morphological opening,
                        the parameter --morph_opening needs to be set. The
                        range is [1, kernel_size] kernel_shape is randomly
                        chosen among [rectangle, ellipse, cross].
  -morphclo MORHP_CLOSING, --morhp_closing MORHP_CLOSING
                        Morphological image processing - closing. The argument
                        must be a tuple separated by comma without space, the
                        first element is the kernel size, the second element
                        is the kernel shape. For example, 3 means
                        kernel_size=3x3. 3,ellipse (3,cross) means using
                        elliptical (cross-shaped) kernel respectively. If the
                        second argument is not given, the default kernel shape
                        will be rectangle.
  -rmorphclo, --random_morph_closing
                        Uniformly sample the value of morphological closing,
                        the parameter --morph_closing needs to be set. The
                        range is [1, kernel_size] kernel_shape is randomly
                        chosen among [rectangle, ellipse, cross].
  -morphgra MORHP_GRADIENT, --morhp_gradient MORHP_GRADIENT
                        Morphological image processing - gradient. The
                        argument must be a tuple separated by comma without
                        space, the first element is the kernel size, the
                        second element is the kernel shape. For example, 3
                        means kernel_size=3x3. 3,ellipse (3,cross) means using
                        elliptical (cross-shaped) kernel respectively. If the
                        second argument is not given, the default kernel shape
                        will be rectangle.
  -rmorphgra, --random_morph_gradient
                        Uniformly sample the value of morphological gradient,
                        the parameter --morph_gradient needs to be set. The
                        range is [1, kernel_size] kernel_shape is randomly
                        chosen among [rectangle, ellipse, cross].
  -morphtoph MORHP_TOPHAT, --morhp_tophat MORHP_TOPHAT
                        Morphological image processing - Top Hat. The argument
                        must be a tuple separated by comma without space, the
                        first element is the kernel size, the second element
                        is the kernel shape. For example, 3 means
                        kernel_size=3x3. 3,ellipse (3,cross) means using
                        elliptical (cross-shaped) kernel respectively. If the
                        second argument is not given, the default kernel shape
                        will be rectangle.
  -rmorphtoph, --random_morph_tophat
                        Uniformly sample the value of morphological tophat,
                        the parameter --morph_tophat needs to be set. The
                        range is [1, kernel_size] kernel_shape is randomly
                        chosen among [rectangle, ellipse, cross].
  -morphblah MORHP_BLACKHAT, --morhp_blackhat MORHP_BLACKHAT
                        Morphological image processing - Black Hat. The
                        argument must be a tuple separated by comma without
                        space, the first element is the kernel size, the
                        second element is the kernel shape. For example, 3
                        means kernel_size=3x3. 3,ellipse (3,cross) means using
                        elliptical (cross-shaped) kernel respectively. If the
                        second argument is not given, the default kernel shape
                        will be rectangle.
  -rmorphblah, --random_morph_blackhat
                        Uniformly sample the value of morphological blackhat,
                        the parameter --morph_blackhat needs to be set. The
                        range is [1, kernel_size] kernel_shape is randomly
                        chosen among [rectangle, ellipse, cross].
```



### Examples

`python3 run.py -c 1000 --size 64`


By default, generated images will be stored to `out/` in the current working directory.

### Text rotation

Add `-rtn` and `-rrtn` (`python3 run.py -c 1000 --size 64 -rtn 5 -rrtn`)


### Text blurring

Add `-bl` and `-rbl` to get gaussian blur on the generated image with user-defined radius (here 0, 1, 2, 4):


### Background



## Add new fonts



## Benchmarks

Number of images generated per second.

Test command: 
```python
python3 run.py --count 1000 --size 32 --ensure_square_layout --image_mode L --dict alphabets/fine/basic_latin_lowercase --font_index fonts/basic_latin_lowercase
```

- 2.7 GHz Dual-Core Intel Core i5 + SSD 
    - `--nb_processes 1` 120 img/s
    - `--nb_processes 2` 152 img/s
    - `--nb_processes 4` 203 img/s
    - `--nb_processes 8` 164 img/s 
    - `--nb_processes 16` 131 img/s 




## Feature request & issues

If anything is missing, unclear, or simply not working, open an issue on the repository.

## Acknowledgement

This project was a fork of [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator). We would like to warmly thank all the contributors of this open source software, especially Edouard Belval. 




