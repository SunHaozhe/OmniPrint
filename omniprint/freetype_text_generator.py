import math 
import numpy as np
from freetype import *
from PIL import Image 
from numba import jit 
from utils import fill_stroke_color


_lt_keywords = ["rotation", "shear_x", "shear_y", "scale_x", "scale_y"]
_lt_keywords += ["alpha", "beta", "gamma", "delta"]
_default_1_keywords = {"scale_x": 3, "scale_y": 4, "alpha": 5, "delta": 8}
_warning_square_layout = "Set margins[0] + margins[2] == margins[1] + margins[3] "
_warning_square_layout += "if you want to ensure square layout."

def param_dict2list(param_dict):
    ''' 
    param_dict: dict
    ''' 
    if param_dict is None or len(param_dict) == 0:
        return [1, 0, 0, 1]
    # normalize keywords 
    for keyword in param_dict:
        if len(keyword) >= 2 and keyword[-1] == "_" and keyword[-2] != "_":
            param_dict[keyword[:-1]] = param_dict.pop(keyword)
    
    res = [] 
    if "a" in param_dict or "b" in param_dict or "d" in param_dict or "e" in param_dict:
        # low level parameterization
        for keyword in ["a", "b", "d", "e"]:
            if keyword in param_dict:
                res.append(param_dict[keyword])
            elif keyword in ["a", "e"]:
                res.append(1)
            else:
                res.append(0)
    else:
        # high level parameterization 
        for keyword in _lt_keywords: 
            if keyword in param_dict:
                res.append(param_dict[keyword])
            else:
                res.append(0) 
        for key, value in _default_1_keywords.items():
            if key not in param_dict:
                res[value] = 1 
    return res


def linear_transform_parameter_formatter(rotation=0, 
                                         shear_x=0, 
                                         shear_y=0, 
                                         scale_x=1, 
                                         scale_y=1,
                                         alpha_=1,
                                         beta_=0, 
                                         gamma_=0,
                                         delta_=1):
    '''
    rotation : float
        Rotation in degrees
    '''
    # convert degree to radian  
    rotation = (rotation / 180.0) * math.pi 
    cos = math.cos(rotation)
    sin = math.sin(rotation)
    
    a_ = scale_x * alpha_ * cos + scale_x * gamma_ * shear_x * cos 
    a_ += - scale_x * alpha_ * shear_y * sin - scale_x * gamma_ * sin
    
    b_ = scale_y * beta_ * cos + scale_y * delta_ * shear_x * cos 
    b_ += - scale_y * beta_ * shear_y * sin - scale_y * delta_ * sin 
    
    d_ = scale_x * alpha_ * sin + scale_x * gamma_ * shear_x * sin
    d_ += scale_x * alpha_ * shear_y * cos + scale_x * gamma_ * cos
    
    e_ = scale_y * beta_ * sin + scale_y * delta_ * shear_x * sin 
    e_ += scale_y * beta_ * shear_y * cos + scale_y * delta_ * cos 
    
    return a_, b_, d_, e_ 


@jit(nopython=True)
def _fill_data(bitmap_buffer, rows, width, pitch):
    data = []
    for j in range(rows):
        data.extend(bitmap_buffer[j * pitch : j * pitch + width])
    return data 


def render_lt_text(text, font_file_path, transform_param=None, 
                   font_size=192, font_weight=None, stroke_radius=None):
    '''
    Adapted from FreeType-py's example code 
    
    render linear transformed text 
    
    Parameters:
    -----------
    text : string
        Text to be displayed
    filename : string
        Path to a font
    transform_param: tuple 
        linear transformation parameters, 4 real numbers
    size : int
        Font size in 1/64th points
        
    TODO:
        stroke_radius not functional yet, Stroker needs debug 
    ''' 
    if isinstance(transform_param, dict):
        transform_param = param_dict2list(transform_param) 
    if transform_param is None:
        # identity transform 
        a_, b_, d_, e_ = 1, 0, 0, 1  
    elif not hasattr(transform_param, "__len__"):
        # rotation only 
        a_, b_, d_, e_ = linear_transform_parameter_formatter(transform_param, 0, 0, 1, 1)
    elif len(transform_param) == 0:
        # identity transform 
        a_, b_, d_, e_ = 1, 0, 0, 1 
    elif len(transform_param) == 1:
        # rotation only 
        a_, b_, d_, e_ = linear_transform_parameter_formatter(transform_param[0], 0, 0, 1, 1)
    elif len(transform_param) == 4:
        # a_, b_, d_, e_, low level parameterization
        a_, b_, d_, e_ = transform_param
    elif len(transform_param) == 5 or len(transform_param) == 9:
        # rotation, shear_x, shear_y, scale_x, scale_y, alpha_, beta_, gamma_, delta_, 
        # high level parameterization
        a_, b_, d_, e_ = linear_transform_parameter_formatter(*transform_param)
    else:
        raise Exception("Wrong transform parameter format.")
    
    assert a_ * e_ - b_ * d_ != 0, "Transform is not invertible."
    
    face = Face(font_file_path)
    # 64 == 2^6, 26.6 fixed-point format 
    face.set_char_size(font_size * 64) 
    # set font weight (stroke width) for variable fonts 
    if font_weight is not None and face.has_multiple_masters:
        face.set_var_design_coords((font_weight,)) 
    # 0x10000 == 65536 == 2^16, 16.16 fixed-point format
    matrix = FT_Matrix(int(a_ * 65536), int(b_ * 65536),
                       int(d_ * 65536), int(e_ * 65536))
    pen = FT_Vector(0, 0)
    FT_Set_Transform(face._FT_Face, byref(matrix), byref(pen))
    flags = FT_LOAD_RENDER 
    previous = 0
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    # First pass to compute bbox
    for c in text:
        face.load_char(c, flags)
        kerning = face.get_kerning(previous, c)
        previous = c
        pitch  = face.glyph.bitmap.pitch
        width  = face.glyph.bitmap.width
        rows   = face.glyph.bitmap.rows
        top    = face.glyph.bitmap_top
        left   = face.glyph.bitmap_left
        pen.x += kerning.x
        x0 = (pen.x >> 6) + left
        x1 = x0 + width
        y0 = (pen.y >> 6) - (rows - top)
        y1 = y0 + rows
        xmin, xmax = min(xmin, x0),  max(xmax, x1)
        ymin, ymax = min(ymin, y0), max(ymax, y1)
        pen.x += face.glyph.advance.x
        pen.y += face.glyph.advance.y
    
    L = np.zeros((ymax - ymin, xmax - xmin),dtype=np.ubyte)
    previous = 0
    pen.x, pen.y = 0, 0 
    # Second pass for actual rendering 
    for c in text:
        face.load_char(c, flags)
        kerning = face.get_kerning(previous, c)
        previous = c
        bitmap = face.glyph.bitmap
        pitch  = face.glyph.bitmap.pitch
        width  = face.glyph.bitmap.width
        rows   = face.glyph.bitmap.rows
        top    = face.glyph.bitmap_top
        left   = face.glyph.bitmap_left
        pen.x += kerning.x
        x = (pen.x >> 6) - xmin + left
        y = (pen.y >> 6) - ymin - (rows - top) 
        data = _fill_data(np.array(bitmap.buffer), rows, width, pitch) 
        if len(data):
            Z = np.array(data, dtype=np.ubyte).reshape(rows, width)
            L[y : y + rows, x : x + width] |= Z
        pen.x += face.glyph.advance.x
        pen.y += face.glyph.advance.y 
    
    img = Image.fromarray(np.invert(L)).convert("RGB") 
    return img, Image.fromarray(L)  












