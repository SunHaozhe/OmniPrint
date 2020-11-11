"""
Utility functions
"""

import os
import glob


def load_dict(lang):
    """Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(
        os.path.join(os.path.dirname(__file__), "dicts", lang + ".txt"),
        "r",
        encoding="utf8",
        errors="ignore",
    ) as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
    return lang_dict


def load_fonts(lang):
    """Load all fonts in the fonts directories
    """

    fonts = []  
    if lang in os.listdir(os.path.join(os.path.dirname(__file__), "fonts")):
        lang_directory = "{}".format(lang) 
    else:
        lang_directory = "latin"
    for p in glob.glob(os.path.join(os.path.dirname(__file__), "fonts", lang_directory, "*.ttf")): 
        fonts.append(p)
    return fonts 
