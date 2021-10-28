"""
render_lt_text is inspired by the example code of FreeType-py
* https://github.com/rougier/freetype-py/tree/master/examples 


freetype-py is licensed under the terms of the new or revised BSD license, as
follows:

Copyright (c) 2011-2014, Nicolas P. Rougier
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

Neither the name of the freetype-py Development Team nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


The example code demonstrates how freetype-py works, it is then 
adapted by Haozhe in order to enable various pre-rasterization 
transformations.
"""


import math 
import numpy as np
from freetype import *
from PIL import Image 
from numba import jit 


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
                   font_size=192, font_weight=None, stroke_radius=None, 
                   pre_elastic=None, stretch_ascender=None, 
                   stretch_descender=None):
    '''
    render initial text (after pre-rasterization transformation)
    
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
    pre_elastic: float in the range [0, 1)
        Pre-rasterization elastic transformation. 
        The actual elastic variation range depends on the product of 
        the range of original anchor points and the value of pre_elastic. 
        If this value is too high, the visual effect may not be good. 
        It is recommended that pre_elastic <= 0.04.
        pre_elastic=0.1 can already lead to unrecognizable text. 
    stretch_ascender: int, positive or negative. 
        The effect of hundreds is sometimes invisible. 
    stretch_descender: int, positive or negative.
        The effect of hundreds is sometimes invisible. 
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
    manipulate_anchor_points = False 
    if pre_elastic is not None:
        assert (isinstance(pre_elastic, float) or pre_elastic == 0) \
            and pre_elastic >= 0 and pre_elastic < 1 
        manipulate_anchor_points = True
    if stretch_ascender is not None:
        assert isinstance(stretch_ascender, int) 
        manipulate_anchor_points = True
    if stretch_descender is not None:
        assert isinstance(stretch_descender, int) 
        manipulate_anchor_points = True
    
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
    flags = FT_LOAD_NO_BITMAP # FT_LOAD_RENDER 
    previous = 0
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    delta_xy = []
    # First pass to compute bbox
    for idx, c in enumerate(text):
        face.load_char(c, flags)
        if pre_elastic:
            pxmin, pymin, pxmax, pymax, = get_anchor_point_statistics(c, face)
            elastic_limit = int((pxmax - pxmin + pymax - pymin) / 2 * pre_elastic)
        if manipulate_anchor_points:
            delta_xy.append([])
            for i in range(face.glyph.outline._FT_Outline.n_points):
                x_delta, y_delta = 0, 0
                if stretch_ascender:
                    # stretch ascender
                    if face.glyph.outline._FT_Outline.points[i].y >= face.ascender:
                        y_delta += stretch_ascender
                if stretch_descender:
                    # stretch descender
                    if face.glyph.outline._FT_Outline.points[i].y <= face.descender:
                        y_delta -= stretch_descender 
                if pre_elastic:
                    # pre-rasterization elastic transformation 
                    x_delta += np.random.randint(- elastic_limit, elastic_limit + 1)
                    y_delta += np.random.randint(- elastic_limit, elastic_limit + 1)
                face.glyph.outline._FT_Outline.points[i].x += x_delta
                face.glyph.outline._FT_Outline.points[i].y += y_delta
                delta_xy[idx].append((x_delta, y_delta))
        ret = FT_Render_Glyph(face.glyph._FT_GlyphSlot, FT_RENDER_MODE_NORMAL)
        assert ret == 0, """Cannot render the transformed glyph. Please decrease 
        the value of pre-rasterization transformation."""
        
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
    for idx, c in enumerate(text):
        face.load_char(c, flags)
        if manipulate_anchor_points:
            for i in range(face.glyph.outline._FT_Outline.n_points):
                face.glyph.outline._FT_Outline.points[i].x += delta_xy[idx][i][0]
                face.glyph.outline._FT_Outline.points[i].y += delta_xy[idx][i][1]
        ret = FT_Render_Glyph(face.glyph._FT_GlyphSlot, FT_RENDER_MODE_NORMAL)
        assert ret == 0, """Cannot render the transformed glyph. Please decrease 
        the value of pre-rasterization transformation."""
        
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


def get_anchor_point_statistics(c, face):
    pxmin, pymin, pxmax, pymax = 0, 0, 0, 0
    face.load_char(c, FT_LOAD_NO_BITMAP)
    for x, y in face.glyph.outline.points:
        pxmin = min(pxmin, x)
        pymin = min(pymin, y)
        pxmax = max(pxmax, x)
        pymax = max(pymax, y)
    return pxmin, pymin, pxmax, pymax














