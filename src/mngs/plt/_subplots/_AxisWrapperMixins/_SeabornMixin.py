#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 16:26:18 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_SeabornMixin.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/plt/_subplots/_AxisWrapperMixins/_SeabornMixin.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from functools import wraps

import mngs
import numpy as np
import pandas as pd
import seaborn as sns

from ....plt import ax as ax_module


def sns_copy_doc(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    wrapper.__doc__ = getattr(sns, func.__name__.split("sns_")[-1]).__doc__
    return wrapper


class SeabornMixin:

    def _sns_base(
        self, method_name, *args, track=True, track_obj=None, id=None, **kwargs
    ):
        sns_method_name = method_name.split("sns_")[-1]

        with self._no_tracking():
            sns_plot_fn = getattr(sns, sns_method_name)

            if kwargs.get("hue_colors"):
                kwargs = mngs.gen.alternate_kwarg(
                    kwargs, primary_key="palette", alternate_key="hue_colors"
                )

            self._axis_mpl = sns_plot_fn(ax=self._axis_mpl, *args, **kwargs)

        # Track the plot if required
        track_obj = track_obj if track_obj is not None else args
        # Create a tracked_dict with appropriate structure
        tracked_dict = {
            'data': track_obj,  # Use 'data' key for consistency with formatters
            'args': args        # Keep args for backward compatibility
        }
        self._track(track, id, method_name, tracked_dict, kwargs)

    def _sns_base_xyhue(self, method_name, *args, track=True, id=None, **kwargs):
        """Formats data passed to sns functions with (data=data, x=x, y=y) keyword arguments"""
        df = kwargs.get("data")
        x, y, hue = kwargs.get("x"), kwargs.get("y"), kwargs.get("hue")

        track_obj = self._sns_prepare_xyhue(df, x, y, hue) if df is not None else None
        self._sns_base(
            method_name,
            *args,
            track=track,
            track_obj=track_obj,
            id=id,
            **kwargs,
        )

    def _sns_prepare_xyhue(self, data=None, x=None, y=None, hue=None, **kwargs):
        """Returns obj to track"""
        data = data.reset_index()

        if hue is not None:
            if x is None and y is None:

                return data
            elif x is None:

                agg_dict = {}
                for hh in data[hue].unique():
                    agg_dict[hh] = data.loc[data[hue] == hh, y]
                df = mngs.pd.force_df(agg_dict)
                return df

            elif y is None:

                df = pd.concat(
                    [data.loc[data[hue] == hh, x] for hh in data[hue].unique()],
                    axis=1,
                )
                return df
            else:
                pivoted_data = data.pivot_table(
                    values=y,
                    index=data.index,
                    columns=[x, hue],
                    aggfunc="first",
                )
                pivoted_data.columns = [
                    f"{col[0]}-{col[1]}" for col in pivoted_data.columns
                ]
                return pivoted_data
        else:
            if x is None and y is None:
                return data

            elif x is None:
                return data[[y]]

            elif y is None:
                return data[[x]]

            else:
                return data.pivot_table(
                    values=y, index=data.index, columns=x, aggfunc="first"
                )

    @sns_copy_doc
    def sns_barplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_barplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_boxplot(
        self,
        data=None,
        x=None,
        y=None,
        strip=False,
        track=True,
        id=None,
        **kwargs,
    ):
        self._sns_base_xyhue(
            "sns_boxplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )
        if strip:
            strip_kwargs = kwargs.copy()
            strip_kwargs.pop("notch", None)  # Remove boxplot-specific kwargs
            strip_kwargs.pop("whis", None)
            self.sns_stripplot(
                data=data,
                x=x,
                y=y,
                track=False,
                id=f"{id}_strip",
                **strip_kwargs,
            )

    @sns_copy_doc
    def sns_heatmap(self, *args, xyz=False, track=True, id=None, **kwargs):
        method_name = "sns_heatmap"
        df = args[0]
        if xyz:
            df = mngs.pd.to_xyz(df)
        self._sns_base(method_name, *args, track=track, track_obj=df, id=id, **kwargs)

    @sns_copy_doc
<<<<<<< HEAD
    def sns_histplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_histplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )
=======
    def sns_histplot(
        self, data=None, x=None, y=None, bins=10, align_bins=True, track=True, id=None, **kwargs
    ):
        """
        Plot a histogram using seaborn with bin alignment support.
        
        This method enhances the seaborn histplot with the ability to align
        bins across multiple histograms on the same axis and export bin data.
        
        Args:
            data: Input data (DataFrame, array, or series)
            x: Column name for x-axis data
            y: Column name for y-axis data
            bins: Bin specification (count or edges)
            align_bins: Whether to align bins with other histograms on this axis
            track: Whether to track this operation
            id: Identifier for tracking
            **kwargs: Additional keywords passed to seaborn histplot
        """
        # Method Name for downstream csv exporting
        method_name = "sns_histplot"
        
        # Get the data to plot for bin alignment
        plot_data = None
        if data is not None and x is not None:
            plot_data = data[x].values if hasattr(data, 'columns') and x in data.columns else None
        
        # Get axis and histogram IDs for bin alignment
        axis_id = str(hash(self._axis_mpl))
        hist_id = id if id is not None else str(self.id)
        
        # Calculate range from data if needed
        range_value = kwargs.get('binrange', None)
        
        # Align bins if requested and data is available
        if align_bins and plot_data is not None:
            from ....plt.utils import histogram_bin_manager
            bins_val, range_val = histogram_bin_manager.register_histogram(
                axis_id, hist_id, plot_data, bins, range_value
            )
            
            # Update bins in kwargs
            kwargs['bins'] = bins_val
            
            # Update range in kwargs if it was provided
            if range_value is not None:
                kwargs['binrange'] = range_val
        
        # Plotting with seaborn
        with self._no_tracking():
            # Execute the seaborn histplot function
            # Don't pass bins as a separate parameter since it may already be in kwargs
            sns_plot = sns.histplot(data=data, x=x, y=y, ax=self._axis_mpl, **kwargs)
            
            # Extract bin information from the plotted artists
            hist_result = None
            if hasattr(sns_plot, 'patches') and sns_plot.patches:
                # Get bin information from the plot patches
                patches = sns_plot.patches
                if patches:
                    counts = np.array([p.get_height() for p in patches])
                    # Extract bin edges from patch positions
                    bin_edges = []
                    for p in patches:
                        bin_edges.append(p.get_x())
                    # Add the rightmost edge
                    if patches:
                        bin_edges.append(patches[-1].get_x() + patches[-1].get_width())
                        
                    hist_result = (counts, np.array(bin_edges))
        
        # Create a track object for the formatter
        track_obj = self._sns_prepare_xyhue(data, x, y, kwargs.get("hue"))
        
        # Enhanced tracked dict with histogram result
        tracked_dict = {
            'data': track_obj,
            'args': (data, x, y),
            'hist_result': hist_result
        }
        
        # Track the operation
        self._track(track, id, method_name, tracked_dict, kwargs)
        
        return sns_plot
>>>>>>> origin/main

    @sns_copy_doc
    def sns_kdeplot(
        self,
        data=None,
        x=None,
        y=None,
        xlim=None,
        ylim=None,
        track=True,
        id=None,
        **kwargs,
    ):
        if kwargs.get("hue"):
            hues = data[kwargs["hue"]]

            if x is not None:
                lim = xlim
                for hue in np.unique(hues):
                    _data = data.loc[hues == hue, x]
                    self.plot_kde(_data, xlim=lim, label=hue, id=hue, **kwargs)

            if y is not None:
                lim = ylim
                for hue in np.unique(hues):
                    _data = data.loc[hues == hue, y]
                    self.plot_kde(_data, xlim=lim, label=hue, id=hue, **kwargs)

        else:
            if x is not None:
                _data, lim = data[x], xlim
            if y is not None:
                _data, lim = data[y], ylim
            self.plot_kde(_data, xlim=lim, **kwargs)

    @sns_copy_doc
    def sns_pairplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_pairplot", *args, track=track, id=id, **kwargs)

    @sns_copy_doc
    def sns_scatterplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_scatterplot",
            data=data,
            x=x,
            y=y,
            track=track,
            id=id,
            **kwargs,
        )

    @sns_copy_doc
    def sns_lineplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_lineplot",
            data=data,
            x=x,
            y=y,
            track=track,
            id=id,
            **kwargs,
        )

    @sns_copy_doc
    def sns_swarmplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_swarmplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_stripplot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
        self._sns_base_xyhue(
            "sns_stripplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    # @sns_copy_doc
    # def sns_violinplot(
    #     self, data=None, x=None, y=None, track=True, id=None, **kwargs
    # ):
    #     self._sns_base_xyhue(
    #         "sns_violinplot", data=data, x=x, y=y, track=track, id=id, **kwargs
    #     )

    @sns_copy_doc
    def sns_violinplot(
        self,
        data=None,
        x=None,
        y=None,
        track=True,
        id=None,
        half=False,
        **kwargs,
    ):
        if half:
            with self._no_tracking():
                self._axis_mpl = ax_module.plot_half_violin(
                    self._axis_mpl, data=data, x=x, y=y, **kwargs
                )
        else:
            self._sns_base_xyhue(
                "sns_violinplot",
                data=data,
                x=x,
                y=y,
                track=track,
                id=id,
                **kwargs,
            )

        # Tracking
        track_obj = self._sns_prepare_xyhue(data, x, y, kwargs.get("hue"))
        self._track(track, id, "sns_violinplot", track_obj, kwargs)

        return self._axis_mpl

    @sns_copy_doc
    def sns_jointplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_jointplot", *args, track=track, id=id, **kwargs)

<<<<<<< HEAD
    def sns_plot(self, plot_type, *args, data=None, x=None, y=None, hue=None, 
                 track=True, id=None, **kwargs):
        """
        General-purpose Seaborn plotting method.
        
        Parameters
        ----------
        plot_type : str
            The name of the Seaborn plotting function (e.g., 'barplot', 'scatterplot')
        *args : tuple
            Positional arguments to pass to the Seaborn function
        data : DataFrame, optional
            Input data structure
        x, y : str, optional
            Column names for x and y axes
        hue : str, optional
            Column name for color grouping
        track : bool, default=True
            Whether to track this plot for export
        id : str, optional
            Identifier for tracking
        **kwargs : dict
            Additional keyword arguments for the Seaborn function
            
        Returns
        -------
        ax : matplotlib Axes
            The axis with the plot
            
        Examples
        --------
        >>> ax.sns_plot('barplot', data=df, x='category', y='value')
        >>> ax.sns_plot('heatmap', correlation_matrix, annot=True)
        >>> ax.sns_plot('regplot', data=df, x='x', y='y', scatter_kws={'alpha': 0.5})
        """
        # Check if the plot type exists in seaborn
        if not hasattr(sns, plot_type):
            raise ValueError(f"'{plot_type}' is not a valid Seaborn plotting function")
        
        # Determine if this is a data-based plot (uses x, y, hue)
        data_based_plots = {
            'barplot', 'boxplot', 'violinplot', 'stripplot', 'swarmplot',
            'pointplot', 'scatterplot', 'lineplot', 'histplot', 'kdeplot',
            'ecdfplot', 'rugplot', 'boxenplot'
        }
        
        # Route to appropriate method
        if plot_type in data_based_plots and data is not None:
            # Use the xyhue method for data-based plots
            method_name = f"sns_{plot_type}"
            self._sns_base_xyhue(
                method_name, 
                data=data, 
                x=x, 
                y=y, 
                hue=hue,
                track=track, 
                id=id, 
                **kwargs
            )
        else:
            # Use the base method for matrix plots or non-data plots
            method_name = f"sns_{plot_type}"
            self._sns_base(
                method_name, 
                *args, 
                track=track, 
                id=id, 
                **kwargs
            )
        
        return self._axis_mpl


=======
>>>>>>> origin/main
# EOF
