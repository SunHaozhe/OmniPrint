# TextRecognitionDataGenerator [![TravisCI](https://travis-ci.org/Belval/TextRecognitionDataGenerator.svg?branch=master)](https://travis-ci.org/Belval/TextRecognitionDataGenerator) [![PyPI version](https://badge.fury.io/py/trdg.svg)](https://badge.fury.io/py/trdg) [![codecov](https://codecov.io/gh/Belval/TextRecognitionDataGenerator/branch/master/graph/badge.svg)](https://codecov.io/gh/Belval/TextRecognitionDataGenerator) [![Documentation Status](https://readthedocs.org/projects/textrecognitiondatagenerator/badge/?version=latest)](https://textrecognitiondatagenerator.readthedocs.io/en/latest/?badge=latest) [![mattermost](https://img.shields.io/badge/help-mattermost-blue)](https://mattermost.belval.org/signup_user_complete/?id=6j1pj6itd7y4781o1u813796ry)

A synthetic data generator for text recognition

## What is it for?

Generating text image samples to train an OCR software. Now supporting non-latin text! For a more thorough tutorial see [the official documentation](https://textrecognitiondatagenerator.readthedocs.io/en/latest/index.html).

## What do I need to make it work?

Install the pypi package

```
pip install trdg
```

Afterwards, you can use `trdg` from the CLI. I recommend using a virtualenv instead of installing with `sudo`.

If you want to add another language, you can clone the repository instead. Simply run `pip install -r requirements.txt`

## Docker image

If you would rather not have to install anything to use TextRecognitionDataGenerator, you can pull the docker image.

```
docker pull belval/trdg:latest

docker run -v /output/path/:/app/out/ -t belval/trdg:latest trdg [args]
```

The path (`/output/path/`) must be absolute.

## New
- Add `--word_split` argument to split on word instead of per-character. This is useful for ligature-based languages
- Add `--dict` argument to specify a custom dictionary (Thank you [@luh0907](https://github.com/luh0907))
- Add `--font_dir` argument to specify the fonts to use
- Add `--output_mask` to output character-level mask for each image
- Add `--character_spacing` to control space between characters (in pixels)
- Add python module
- Add `--font` to use only one font for all the generated images (Thank you [@JulienCoutault](https://github.com/JulienCoutault)!)
- Add `--fit` and `--margins` for finer layout control
- Change the text orientation using the `-or` parameter
- Specify text color range using `-tc '#000000,#FFFFFF'`, please note that the quotes are **necessary**
- Add support for Simplified and Traditional Chinese

## How does it work?

```zsh
python3 quick_run.py  
```

```zsh
usage: run.py [-h] [--output_dir [OUTPUT_DIR]] [-i [INPUT_FILE]]
              [-l [LANGUAGE]] -c [COUNT] [-rs] [-let] [-num] [-sym]
              [-w [LENGTH]] [-r] [-s [SIZE]] [-p [NB_PROCESSES]]
              [-e [EXTENSION]] [-wk] [-bl [BLUR]] [-rbl] [-b [BACKGROUND]]
              [-hw] [-na NAME_FORMAT] [-om OUTPUT_MASK] [-d [DISTORSION]]
              [-do [DISTORSION_ORIENTATION]] [-m [MARGINS]] [-ft [FONT]]
              [-fd [FONT_DIR]] [-fidx [FONT_INDEX]] [-id [IMAGE_DIR]]
              [-ca [CASE]] [-dt [DICT]] [-fwt [FONT_WEIGHT]]
              [-stf [STROKE_FILL]] [-im [IMAGE_MODE]] [-rsd RANDOM_SEED]
              [-esl] [-otlwd OUTLINE_WIDTH] [-fsz FONT_SIZE]
              [-lt [LINEAR_TRANSFORM]] [-rtn [ROTATION]] [-rrtn]
              [-shrx [SHEAR_X]] [-rshrx] [-shry [SHEAR_Y]] [-rshry]
              [-sclx [SCALE_X]] [-rsclx] [-scly [SCALE_Y]] [-rscly]
              [-alpha [ALPHA]] [-ralpha] [-beta [BETA]] [-rbeta]
              [-gamma [GAMMA]] [-rgamma] [-delta [DELTA]] [-rdelta] [-rtslnx]
              [-rtslny]

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
                        an integer defining the blur radius
  -rbl, --random_blur   When set, the blur radius will be randomized between 0
                        and -bl.
  -b [BACKGROUND], --background [BACKGROUND]
                        Define what kind of background to use. 0: Gaussian
                        Noise, 1: Plain white, 2: Quasicrystal, 3: Image
  -hw, --handwritten    Define if the data will be "handwritten" by an RNN
  -na NAME_FORMAT, --name_format NAME_FORMAT
                        Define how the produced files will be named. 0:
                        [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT]
                        + one file labels.txt containing id-to-label mappings
  -om OUTPUT_MASK, --output_mask OUTPUT_MASK
                        Define if the generator will return masks for the text
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

```

Words will be randomly chosen from a dictionary of a specific language. Then an image of those words will be generated by using font, background, and modifications (rotation, blurring, etc.) as specified.

### Basic (Python module)

The usage as a Python module is very similar to the CLI, but it is more flexible if you want to include it directly in your training pipeline, and will consume less space and memory. There are 4 generators that can be used.

```py
from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)

# The generators use the same arguments as the CLI, only as parameters
generator = GeneratorFromStrings(
    ['Test1', 'Test2', 'Test3'],
    blur=2,
    random_blur=True
)

for img, lbl in generator:
    # Do something with the pillow images here.
```

You can see the full class definition here:

- [`GeneratorFromDict`](trdg/generators/from_dict.py)
- [`GeneratorFromRandom`](trdg/generators/from_random.py)
- [`GeneratorFromStrings`](trdg/generators/from_strings.py)
- [`GeneratorFromWikipedia`](trdg/generators/from_wikipedia.py)

### Basic (CLI)

`trdg -c 1000 -w 5 -f 64`

You get 1,000 randomly generated images with random text on them like:

![1](samples/1.jpg "1")
![2](samples/2.jpg "2")
![3](samples/3.jpg "3")
![4](samples/4.jpg "4")
![5](samples/5.jpg "5")

By default, they will be generated to `out/` in the current working directory.

### Text skewing

What if you want random skewing? Add `-k` and `-rk` (`trdg -c 1000 -w 5 -f 64 -k 5 -rk`)

![6](samples/6.jpg "6")
![7](samples/7.jpg "7")
![8](samples/8.jpg "8")
![9](samples/9.jpg "9")
![10](samples/10.jpg "10")

### Text distortion
You can also add distorsion to the generated text with `-d` and `-do`

![23](samples/24.jpg "0")
![24](samples/25.jpg "1")
![25](samples/26.jpg "2")

### Text blurring

But scanned document usually aren't that clear are they? Add `-bl` and `-rbl` to get gaussian blur on the generated image with user-defined radius (here 0, 1, 2, 4):

![11](samples/11.jpg "0")
![12](samples/12.jpg "1")
![13](samples/13.jpg "2")
![14](samples/14.jpg "4")

### Background

Maybe you want another background? Add `-b` to define one of the three available backgrounds: gaussian noise (0), plain white (1), quasicrystal (2) or image (3).

![15](samples/15.jpg "0")
![16](samples/16.jpg "1")
![17](samples/17.jpg "2")
![23](samples/23.jpg "3")

When using image background (3). A image from the images/ folder will be randomly selected and the text will be written on it.

### Handwritten

Or maybe you are working on an OCR for handwritten text? Add `-hw`! (Experimental)

![18](samples/18.jpg "0")
![19](samples/19.jpg "1")
![20](samples/20.jpg "2")
![21](samples/21.jpg "3")
![22](samples/22.jpg "4")

It uses a Tensorflow model trained using [this excellent project](https://github.com/Grzego/handwriting-generation) by Grzego.

**The project does not require TensorFlow to run if you aren't using this feature**

### Dictionary

The text is chosen at random in a dictionary file (that can be found in the *dicts* folder) and drawn on a white background made with Gaussian noise. The resulting image is saved as [text]\_[index].jpg

There are a lot of parameters that you can tune to get the results you want, therefore I recommend checking out `trdg -h` for more information.

## Create images with Chinese text

It is simple! Just do `trdg -l cn -c 1000 -w 5`!

Generated texts come both in simplified and traditional Chinese scripts.

Traditional:

![27](samples/27.jpg "0")

Simplified:

![28](samples/28.jpg "1")

## Add new fonts

The script picks a font at random from the *fonts* directory.

| Directory | Languages |
|:----|:-----|
| fonts/latin | English, French, Spanish, German |
| fonts/cn | Chinese |
| fonts/ko | Korean |

Simply add/remove fonts until you get the desired output.

If you want to add a new non-latin language, the amount of work is minimal.

1. Create a new folder with your language [two-letters code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
2. Add a .ttf font in it
3. Edit `run.py` to add an if statement in `load_fonts()`
4. Add a text file in `dicts` with the same two-letters code
5. Run the tool as you normally would but add `-l` with your two-letters code

It only supports .ttf for now.

## Benchmarks

Number of images generated per second.

Test command: 
```python
python3 run.py --count 1000 --size 32 --ensure_square_layout --image_mode L --dict alphabets/fine/basic_latin_lowercase --font_index prepare_fonts/fonts/basic_latin_lowercase
```

- 2.7 GHz Dual-Core Intel Core i5 + SSD 
    - `--nb_processes 1` 120 img/s
    - `--nb_processes 2` 152 img/s
    - `--nb_processes 4` 203 img/s
    - `--nb_processes 8` 164 img/s 
    - `--nb_processes 16` 131 img/s 


## Contributing

1. Create an issue describing the feature you'll be working on
2. Code said feature
3. Create a pull request

## Feature request & issues

If anything is missing, unclear, or simply not working, open an issue on the repository.

## What is left to do?
- Better background generation
- Better handwritten text generation
- More customization parameters (mostly regarding background)
