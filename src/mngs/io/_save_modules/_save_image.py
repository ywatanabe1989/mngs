#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-29 19:42:31 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/io/_save_image.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_image.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import io as _io

import plotly
from PIL import Image


def _save_image(obj, spath, **kwargs):
    import os
    import matplotlib.pyplot as plt
    
    # Process any separate legend requests first
    fig = None
    try:
        # Get the figure object - handle both wrapped and unwrapped figures
        if hasattr(obj, '_fig_mpl'):
            # This is a wrapped figure (FigWrapper)
            fig = obj._fig_mpl
        elif hasattr(obj, 'savefig'):
            # This is a matplotlib figure
            fig = obj
        elif hasattr(obj, 'figure'):
            fig = obj.figure
        elif hasattr(obj, 'get_figure'):
            fig = obj.get_figure()
        
        # Check if there are separate legend params
        if fig and hasattr(fig, '_separate_legend_params'):
            for legend_params in fig._separate_legend_params:
                # Generate legend filename based on main file
                dir_name = os.path.dirname(spath)
                base_name = os.path.basename(spath)
                name_parts = os.path.splitext(base_name)
                ext = name_parts[1].lower()
                # Use axis_id if available for unique legend filenames
                if 'axis_id' in legend_params and legend_params['axis_id']:
                    legend_filename = os.path.join(dir_name, f"{name_parts[0]}_{legend_params['axis_id']}_legend{ext}")
                else:
                    legend_filename = os.path.join(dir_name, f"{name_parts[0]}_legend{ext}")
                
                # Create legend figure
                fig_legend = plt.figure(figsize=legend_params['figsize'])
                fig_legend.legend(
                    legend_params['handles'],
                    legend_params['labels'],
                    loc='center',
                    frameon=legend_params['frameon'],
                    fancybox=legend_params['fancybox'],
                    shadow=legend_params['shadow'],
                    **legend_params['kwargs']
                )
                
                # Save legend - handle GIF format specially
                if ext == '.gif':
                    # For GIF, save as PNG first then convert
                    buf = _io.BytesIO()
                    fig_legend.savefig(buf, format="png", dpi=legend_params['dpi'], bbox_inches='tight')
                    buf.seek(0)
                    img = Image.open(buf)
                    img.save(legend_filename, "GIF")
                    buf.close()
                else:
                    # For other formats, save directly
                    fig_legend.savefig(legend_filename, dpi=legend_params['dpi'], bbox_inches='tight')
                
                plt.close(fig_legend)
                
                print(f"Legend saved to: {legend_filename}")
            
            # Clear the params after processing
            del fig._separate_legend_params
    except Exception as e:
        import logging
        logging.error(f"Error occurred while saving legend: {e}")
        # Don't re-raise to allow main figure to save
    
    # Track the saved filename in the figure wrapper for later use
    try:
        # Check if this is a wrapped figure object
        if hasattr(obj, '_last_saved_info'):
            obj._last_saved_info = spath
        # Check if the object has a figure attribute that's wrapped
        elif hasattr(obj, 'figure'):
            if hasattr(obj.figure, '_last_saved_info'):
                obj.figure._last_saved_info = spath
            else:
                # Set directly on matplotlib figure
                obj.figure._last_saved_info = spath
        # Check if this is an axes object with a wrapped figure
        elif hasattr(obj, 'get_figure'):
            fig = obj.get_figure()
            if fig is not None:
                fig._last_saved_info = spath
        # Check if this is a matplotlib figure directly
        elif hasattr(obj, 'savefig'):
            obj._last_saved_info = spath
    except:
        pass

    # png
    if spath.endswith(".png"):
        # plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(file=spath, format="png")
        # PIL image
        elif isinstance(obj, Image.Image):
            obj.save(spath)
        # matplotlib
        else:
            try:
                obj.savefig(spath, bbox_inches='tight')
            except:
                obj.figure.savefig(spath, bbox_inches='tight')
        del obj

    # tiff
    elif spath.endswith(".tiff") or spath.endswith(".tif"):
        # PIL image
        if isinstance(obj, Image.Image):
            obj.save(spath)
        # matplotlib
        else:
            try:
                obj.savefig(spath, dpi=300, format="tiff", bbox_inches='tight')
            except:
                obj.figure.savefig(spath, dpi=300, format="tiff", bbox_inches='tight')

        del obj

    # jpeg
    elif spath.endswith(".jpeg") or spath.endswith(".jpg"):
        buf = _io.BytesIO()

        # plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(buf, format="png")
            buf.seek(0)
            img = Image.open(buf)
            img.convert("RGB").save(spath, "JPEG")
            buf.close()

        # PIL image
        elif isinstance(obj, Image.Image):
            obj.save(spath)

        # matplotlib
        else:
            try:
                obj.savefig(buf, format="png", bbox_inches='tight')
            except:
                obj.figure.savefig(buf, format="png", bbox_inches='tight')

            buf.seek(0)
            img = Image.open(buf)
            img.convert("RGB").save(spath, "JPEG")
            buf.close()
        del obj

    # SVG
    elif spath.endswith(".svg"):
        # Plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(file=spath, format="svg")
        # Matplotlib
        else:
            try:
                obj.savefig(spath, format="svg", bbox_inches='tight')
            except AttributeError:
                obj.figure.savefig(spath, format="svg", bbox_inches='tight')
        del obj
    
    # GIF
    elif spath.endswith(".gif"):
        # For static images, we convert to GIF via PIL
        buf = _io.BytesIO()
        
        # plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(buf, format="png")
            buf.seek(0)
            img = Image.open(buf)
            img.save(spath, "GIF")
            buf.close()
        
        # PIL image
        elif isinstance(obj, Image.Image):
            obj.save(spath, "GIF")
        
        # matplotlib
        else:
            try:
                obj.savefig(buf, format="png", bbox_inches='tight')
            except:
                obj.figure.savefig(buf, format="png", bbox_inches='tight')
            
            buf.seek(0)
            img = Image.open(buf)
            img.save(spath, "GIF")
            buf.close()
        del obj


# EOF
