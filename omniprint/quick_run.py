"""
Helper module to run "run.py"
"""


class Parameters:
    count = 20                                  # The total number of images to be created.
    size = 128                                  # The height of the produced images
    ensure_square_layout = True                 # Force the width to be the same as the height

    dict_ = "alphabets/fine/ethiopic_syllables" # generate characters from Ethiopic syllables
    image_mode = "RGB"                          # generate RGB images
    margins = "0.08,0.08,0.08,0.08"             # Define the 4 margins (percentage)
    
    pre_elastic = 0.03                          # pre-rasterization elastic transformation
    rotation = "-60 60"                         # random rotaiton within -60 degrees and 60 degrees
    shear_x = "-0.4 0.4"                        # random horizontal shear within -0.4 and 0.4
    random_perspective_transform = 0.08         # random perspective transformation at the level 0.08
    random_translation_x = True                 # random horizontal translation
    random_translation_y = True                 # random vertical translation
    
    foreground_image = True                     # Texture foreground
    outline = "image"                           # Texture text outline
    background = "image"                        # Texture background 
    image_blending_method = "poisson"           # Use Poisson Image Editing to blend foreground and background

    brightness = "0.95 1.05"                    # random brightness within 0.95 and 1.05
    contrast = "0.95 1.05"                      # random contrast within 0.95 and 1.05
    color_enhance = "0.95 1.05"                 # random color enhancement within 0.95 and 1.05



if __name__ == "__main__":
    import subprocess

    ignore_list = ["__module__", "__dict__", "__weakref__", "__doc__"]

    cmd = "python3 run.py"
    
    parameters = vars(Parameters) 
    for key, value in parameters.items():
        if key in ignore_list:
            continue 
        if len(key) >= 2 and key[-1] == "_" and key[-2] != "_":
            key = key[:-1]
        if isinstance(value, bool) and value:
            cmd += " --{}".format(key)
        else:
            cmd += " --{} {}".format(key, str(value))

    # run the command 
    subprocess.call(cmd.split())


    




