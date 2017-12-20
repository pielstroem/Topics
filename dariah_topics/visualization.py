#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizing the Output of LDA Models
************************************

Functions and classes of this module are for visualizing LDA models.

Contents
********
    * 
"""


import logging
from dariah_topics import postprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import pandas as pd
from bokeh.plotting import figure
from bokeh import palettes
from bokeh.models import (
            ColumnDataSource,
            HoverTool,
            LinearColorMapper,
            BasicTicker,
            ColorBar
            )

import regex
from collections import defaultdict
from wordcloud import WordCloud

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
logging.basicConfig(level = logging.ERROR,
                    format = '%(levelname)s %(name)s: %(message)s')


def plot_wordcloud(weights, enable_notebook=True, **kwargs):
    """Plots a wordcloud based on tokens and frequencies.
    
    Args:
        weights (dict): A dictionary (or :module:``pandas`` Series) with tokens
            as keys and frequencies as values.
        enable_notebook (bool), optional: If True, enables :module:``matplotlib``
            to show its figures within a Jupyter notebook.
        font_path (str), optional: Font path to the font that will be used (OTF or TTF).
            Defaults to DroidSansMono path on a Linux machine. If you are on
            another OS or don't have this font, you need to adjust this path.
        width (int), optional: Width of the canvas. Defaults to 400.
        height (int), optional: Height of the canvas. Defaults to 200.
        prefer_horizontal (float): The ratio of times to try horizontal fitting
            as opposed to vertical. If ``prefer_horizontal < 1``, the algorithm
            will try rotating the word if it doesn't fit. (There is currently no
            built-in way to get only vertical words. Defaults to 0.90.
        mask (nd-array), optional: If not None, gives a binary mask on where to draw words.
            If mask is not None, width and height will be ignored and the shape
            of mask will be used instead. All white (#FF or #FFFFFF) entries
            will be considerd 'masked out' while other entries will be free to
            draw on. Defaults to None.
        scale (float), optional: Scaling between computation and drawing. For large word-cloud
            images, using scale instead of larger canvas size is significantly
            faster, but might lead to a coarser fit for the words. Defaults to 1.
        min_font_size (int), optional: Smallest font size to use. Will stop when there is
            no more room in this size. Defaults to 4.
        font_step (int), optional: Step size for the font. ``font_step > 1`` might speed
            up computation but give a worse fit. Defaults to 1.
        max_words (int), optional: The maximum number of words. Defaults to 200.
        stopwords (set), optional: The words that will be eliminated. If None, the build-in
            stopwords list will be used.
        background_color (str), optional: Background color for the word cloud image.
            Defaults to ``black``.
        max_font_size (int), optional: Maximum font size for the largest word. If None,
            height of the image is used.
        mode (str), optional: Transparent background will be generated when mode is ``RGBA``
            and background_color is None. Defaults to ``RGB``.
        relative_scaling (float), optional: Importance of relative word frequencies for
            font-size. With ``relative_scaling=0``, only word-ranks are considered.
            With ``relative_scaling=1``, a word that is twice as frequent will
            have twice the size. If you want to consider the word frequencies and
            not only their rank, ``relative_scaling`` around .5 often looks good.
            Defaults to 0.5.
        color_func (callable), optional: Callable with parameters ``word``, ``font_size``,
            ``position``, ``orientation``, ``font_path``, ``random_state`` that
            returns a PIL color for each word. Overwrites ``colormap``. See ``colormap``
            for specifying a :module:``matplotlib`` colormap instead.
        collocations (bool), optional: Whether to include collocations (bigrams) of two words.
            Defaults to True.
        colormap (str), optional: :module:``matplotlib`` colormap to randomly draw colors
            from for each word. Ignored if ``color_func`` is specified. Defaults to
            ``viridis``.
        normalize_plurals (bool), optional: Whether to remove trailing 's' from words. If
            True and a word appears with and without a trailing 's', the one with
            trailing 's' is removed and its counts are added to the version without
            trailing 's' -- unless the word ends with 'ss'. Defaults to True.

    Returns:
        WordCloud object.
        
    Example:
        >>> weights = {'an': 2, 'example': 1}
        >>> plot_wordcloud(weights, enable_notebook=False) # doctest: +ELLIPSIS
        <wordcloud.wordcloud.WordCloud object at ...>
    """
    wordcloud = WordCloud(**kwargs).fit_words(weights)
    if enable_notebook:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        try:
            fig, ax = plt.subplots(figsize=(kwargs['width'] / 96, kwargs['height'] / 96))
        except KeyError:
            fig, ax = plt.subplots(figsize=(400 / 96, 200 / 96))
        ax.axis('off')
        ax.imshow(wordcloud)
    return wordcloud


def plot_key_frequencies(keys=None, overall_freqs=None, within_topic_freqs=None,
                         within_topic_color='#FF1727', document_term_matrix=None,
                         model=None, vocabulary=None, topic_no=None, overall_color='#053967',
                         figsize=(15, 7), dpi=None, overall_edgecolor=None,
                         overall_linewidth=None, overall_alpha=0.9, within_topic_edgecolor=None,
                         within_topic_linewidth=None, within_topic_alpha=0.9,
                         label_fontsize=15, num_keys=None, tick_fontsize=14, legend_fontsize=15,
                         legend=True, enable_notebook=True):
    """Plots key frequencies overall and from within topic.
    
    Args:
        keys (list): A list of tokens. Defaults to None.
        overall_freqs (list): A list of frequencies. Defaults to None.
        within_topic_freqs (list): A list of frequencies. Defaults to None.
        within_topic_color (str), optional: Color for topic frequencies bar. Defaults to
            ``#FF1727``.
        document_term_matrix (pandas DataFrame), optional: A document-term matrix. Defaults
            to None.
        model, optional: A LDA model. Defaults to None.
        vocabulary (list), optional: Vocabulary of the corpus. Defaults to None.
        topic_no (int), optional: Number of topic. Defaults to None.
        overall_color (str), optional: Color for overall frequencies bar. Defaults to ``#053967``.
        figsize (tuple), optional: Size of the figure. Defaults to ``(15, 7)``.
        dpi (int), optional: Dots per inch. Defaults to None.
        overall_edgecolor (str), optional: Color for edgecolors of overall frequencies bar.
            Defaults to None.
        overall_linewidth (int), optional: Linewidth of overall frequencies bar. Defaults to
            None.
        overall_alpha (int), optional: Alpha for overall frequencies bar. Defaults to 0.9.
        within_topic_edgecolor (str), optional: Color for edgecolors of overall frequencies bar.
            Defaults to None.
        within_topic_linewidth (int), optional: Linewidth of overall frequencies bar. Defaults to
            None.
        within_topic_alpha (int), optional: Alpha for overall frequencies bar. Defaults to 0.9.
        label_fontsize (int), optional: Fontsize of x-axis and y-axis labels. Defaults to 15.
        num_keys (int), optional: Number of tokens for y-axis. Defaults to None.
        tick_fontsize (int), optional: Fontsize of x- and y-ticks. Defaults to 14.
        legend_fontsize (int), optional: Fontsize of the legend. Defaults to 15.
        legend (bool), optional: If True, legend will be displayed. Defaults to True.
        enable_notebook (bool), optional: If True, enables :module:``matplotlib``
            to show its figures within a Jupyter notebook.

    Returns:
        Figure object.
        
    Example:
        >>> keys = ['one', 'example']
        >>> overall_freqs = [20, 10]
        >>> within_topic_freqs = [10, 5]
        >>> plot_key_frequencies(keys=keys,
        ...                      overall_freqs=overall_freqs,
        ...                      within_topic_freqs=within_topic_freqs,
        ...                      enable_notebook=False) # doctest: +ELLIPSIS
        <matplotlib.figure.Figure object at ...>
    """
    if enable_notebook:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
    if model:
        within_topic_freqs = postprocessing.get_sorted_values_from_distribution(model.components_[topic_no],
                                                                                model.components_[topic_no],
                                                                                num_keys)
        within_topic_freqs = [dist * len(vocabulary) for dist in within_topic_freqs]
        total = [document_term_matrix[token].sum() for token in vocabulary]
        overall_freqs = postprocessing.get_sorted_values_from_distribution(total,
                                                                           model.components_[topic_no],
                                                                           num_keys)
        keys = postprocessing.get_sorted_values_from_distribution(vocabulary,
                                                                  model.components_[topic_no],
                                                                  num_keys)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y_axis = np.arange(len(keys))
    overall = ax.barh(y_axis, overall_freqs, color=overall_color, edgecolor=overall_edgecolor,
                      linewidth=overall_linewidth, alpha=overall_alpha)
    within = ax.barh(y_axis, within_topic_freqs, color=within_topic_color,
                     edgecolor=within_topic_edgecolor, linewidth=within_topic_linewidth,
                     alpha=within_topic_alpha)
    ax.set_xlabel('Frequency', fontsize=label_fontsize)
    ax.set_ylabel('Key', fontsize=label_fontsize)
    ax.set_yticks(y_axis)
    ax.set_yticklabels(keys, fontsize=tick_fontsize)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    if legend:
        ax.legend(handles=[overall, within], labels=['Overall', 'Within Topic'], loc='best',
                  fontsize=legend_fontsize)
    return fig


class PlotDocumentTopics:
    """
    Class to visualize document-topic matrix.
    """
    def __init__(self, document_topics, enable_notebook=True):
        self.document_topics = document_topics
        if enable_notebook:
            self.enable_notebook = enable_notebook
            self.show = self.notebook_handling()

    @staticmethod
    def notebook_handling():
        """Runs cell magic for Jupyter notebooks
        """
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        from bokeh.io import output_notebook, show
        output_notebook()
        return show

    def static_heatmap(self, figsize=(1000 / 96, 600 / 96), dpi=None,
                       labels_fontsize=13, cmap='Blues', ticks_fontsize=12,
                       xlabel='Document', ylabel='Topic', xticks_bottom=0.1,
                       xticks_rotation=50, xticks_ha='right', colorbar=False):
        """Plots a static heatmap.
    
        Args:
            figsize (tuple), optional: Size of the figure in inches. Defaults to
                ``(1000 / 96, 500 / 96)``.
            dpi (int), optional: Dots per inch. Defaults to None.
            labels_fontsize (int), optional: Fontsize of the figure labels. Defaults
                to 13.
            cmap (str), optional: Colormap for the figure. Defaults to ``Blues``.
            ticks_fontsize (int), optional: Fontsize of axis ticks. Defaults to 12.
            xlabel (str), optional: Label of x-axis. Defaults to ``Document``.
            ylabel (str), optional: Label of y-axis. Defaults to ``Topic``.
            xticks_bottom (str), optional: Distance to bottom of x-ticks. Defaults
                to 0.1.
            xticks_rotation (int), optional: Rotation degree of x-ticks. Defaults
                to 50.
            xticks_ha (str), optional: The horizontal alignment of the x-tick labels.
                Defaulst to ``right``.
            colorbar (bool), optional: If True, include colorbar. Defaults to True.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        heatmap = ax.pcolor(self.document_topics, cmap=cmap)
        ax.set_xlabel(xlabel, fontsize=labels_fontsize)
        ax.set_ylabel(ylabel, fontsize=labels_fontsize)
        ax.set_xticks(np.arange(self.document_topics.shape[1]) + 0.5)
        ax.set_yticks(np.arange(self.document_topics.shape[0]) + 0.5)
        ax.set_xticklabels(list(self.document_topics.columns), fontsize=ticks_fontsize)
        ax.set_yticklabels(list(self.document_topics.index), fontsize=ticks_fontsize)
        fig.autofmt_xdate(bottom=xticks_bottom, rotation=xticks_rotation, ha=xticks_ha)
        if colorbar:
            cax = ax.imshow(self.document_topics, interpolation='nearest', cmap=cmap)
            cbar = fig.colorbar(cax, ticks=np.arange(0, 1, 0.1))
        return fig
        
    def __static_barchart(self, index, describer, figsize=(11, 7), color='#053967',
                          edgecolor=None, linewidth=None, alpha=None, labels_fontsize=15,
                          ticks_fontsize=14, title=True, title_fontsize=17,
                          dpi=None, transpose_data=False):
        """Plots a static barchart.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            title_fontsize (int), optional: Fontsize of figure title.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            color (str), optional: Color of the bins. Defaults to ``#053967``.
            edgecolor (str), optional: Color of the bin edges. Defaults to None.
            lindewidth (float), optional: Width of bin lines. Defaults to None.
            alpha (float): Alpha value used for blending. Defaults to None.
            figsize (tuple), optional: Size of the figure in inches. Defaults to
                ``(1000 / 96, 500 / 96)``.
            dpi (int), optional: Dots per inch. Defaults to None.
            labels_fontsize (int), optional: Fontsize of the figure labels. Defaults
                to 15.
            ticks_fontsize (int), optional: Fontsize of axis ticks. Defaults to 14.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi) 
        if isinstance(index, int):
            if transpose_data:
                proportions = self.document_topics.T.iloc[index]
            else:
                proportions = self.document_topics.iloc[index]
            if title:
                plot_title = '{0}: {1}'.format(describer, proportions.name)
                ax.set_title(plot_title, fontsize=title_fontsize)
        elif isinstance(index, str):
            if transpose:
                proportions = self.document_topics.T.loc[index]
            else:
                proportions = self.document_topics.loc[index]
            if title:
                plot_title = '{}: {}'.format(describer, index)
                ax.set_title(plot_title, fontsize=title_fontsize)
        else:
            raise ValueError("{} must be int or str.".format(index))
        
        y_axis = np.arange(len(proportions))
        x_axis = proportions
        y_ticks_labels = proportions.index
        ax.barh(y_axis, x_axis, color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
        ax.set_xlabel('Proportion', fontsize=labels_fontsize)
        ax.set_ylabel(describer, fontsize=labels_fontsize)
        ax.set_yticks(y_axis)
        ax.set_yticklabels(y_ticks_labels, fontsize=ticks_fontsize)
        ax.tick_params(axis='x', labelsize=ticks_fontsize)
        return fig

    def static_barchart_per_topic(self, **kwargs):
        """Plots a static barchart per topic.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            title_fontsize (int), optional: Fontsize of figure title.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            color (str), optional: Color of the bins. Defaults to ``#053967``.
            edgecolor (str), optional: Color of the bin edges. Defaults to None.
            lindewidth (float), optional: Width of bin lines. Defaults to None.
            alpha (float): Alpha value used for blending. Defaults to None.
            figsize (tuple), optional: Size of the figure in inches. Defaults to
                ``(1000 / 96, 500 / 96)``.
            dpi (int), optional: Dots per inch. Defaults to None.
            labels_fontsize (int), optional: Fontsize of the figure labels. Defaults
                to 15.
            ticks_fontsize (int), optional: Fontsize of axis ticks. Defaults to 14.

        Returns:
            Figure object.
        """
        return self.__static_barchart(**kwargs)
        
    def static_barchart_per_document(self, **kwargs):
        """Plots a static barchart per document.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            title_fontsize (int), optional: Fontsize of figure title.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            color (str), optional: Color of the bins. Defaults to ``#053967``.
            edgecolor (str), optional: Color of the bin edges. Defaults to None.
            lindewidth (float), optional: Width of bin lines. Defaults to None.
            alpha (float): Alpha value used for blending. Defaults to None.
            figsize (tuple), optional: Size of the figure in inches. Defaults to
                ``(1000 / 96, 500 / 96)``.
            dpi (int), optional: Dots per inch. Defaults to None.
            labels_fontsize (int), optional: Fontsize of the figure labels. Defaults
                to 15.
            ticks_fontsize (int), optional: Fontsize of axis ticks. Defaults to 14.

        Returns:
            Figure object.
        """
        return self.__static_barchart(**kwargs, transpose_data=True)

    def interactive_heatmap(self, palette=palettes.Blues[9], reverse_palette=True,
                            tools='hover, pan, reset, save, wheel_zoom, zoom_in, zoom_out',
                            width=1000, height=550, x_axis_location='below', toolbar_location='above',
                            sizing_mode='fixed', line_color=None, grid_line_color=None, axis_line_color=None,
                            major_tick_line_color=None, major_label_text_font_size='9pt',
                            major_label_standoff=0, major_label_orientation=3.14/3, colorbar=True):
        """Plots an interactive heatmap.
    
        Args:
            palette (list), optional: A list of color values. Defaults to ``palettes.Blues[9]``.
            reverse_palette (bool), optional: If True, color values of ``palette`` will
                be reversed. Defaults to True.
            tools (str), optional: Tools, which will be includeded. Defaults to ``hover,
                pan, reset, save, wheel_zoom, zoom_in, zoom_out``.
            width (int), optional: Width of the figure. Defaults to 1000.
            height (int), optional: Height of the figure. Defaults to 550.
            x_axis_location (str), optional: Location of the x-axis. Defaults to
                ``below``.
            toolbar_location (str), optional: Location of the toolbar. Defaults to
                ``above``.
            sizing_mode (str), optional: Size fixed or width oriented. Defaults to ``fixed``.
            line_color (str): Color for lines. Defaults to None.
            grid_line_color (str): Color for grid lines. Defaults to None.
            axis_line_color (str): Color for axis lines. Defaults to None.
            major_tick_line_color (str): Color for major tick lines. Defaults to None.
            major_label_text_font_size (str): Font size for major label text. Defaults
                to ``9pt``.
            major_label_standoff (int): Standoff for major labels. Defaults to 0.
            major_label_orientation (float): Orientation for major labels. Defaults
                to ``3.14 / 3``.
            colorbar (bool): If True, colorbar will be included.

        Returns:
            Figure object.
        """        
        if reverse_palette:
            palette = list(reversed(palette))

        x_range = list(self.document_topics.columns)
        y_range = list(self.document_topics.index)
        
        stacked_data = pd.DataFrame(self.document_topics.stack()).reset_index()
        stacked_data.columns = ['Topics', 'Documents', 'Distributions']
        mapper = LinearColorMapper(palette=palette,
                                   low=stacked_data.Distributions.min(),
                                   high=stacked_data.Distributions.max())
        source = ColumnDataSource(stacked_data)
        
        fig = figure(x_range=x_range,
                     y_range=y_range,
                     x_axis_location=x_axis_location,
                     plot_width=width, plot_height=height,
                     tools=tools, toolbar_location=toolbar_location,
                     sizing_mode=sizing_mode,
                     logo=None)
        fig.rect(x='Documents', y='Topics', source=source, width=1, height=1,
                 fill_color={'field': 'Distributions', 'transform': mapper},
                 line_color=line_color)

        fig.grid.grid_line_color = grid_line_color
        fig.axis.axis_line_color = axis_line_color
        fig.axis.major_tick_line_color = major_tick_line_color
        fig.axis.major_label_text_font_size = major_label_text_font_size
        fig.axis.major_label_standoff = major_label_standoff
        fig.xaxis.major_label_orientation = major_label_orientation
        
        if 'hover' in tools:
            fig.select_one(HoverTool).tooltips = [('Document', '@Documents'),
                                                  ('Topic', '@Topics'),
                                                  ('Score', '@Distributions')]

        if colorbar:
            feature = ColorBar(color_mapper=mapper, major_label_text_font_size=major_label_text_font_size,
                               ticker=BasicTicker(desired_num_ticks=len(palette)),
                               label_standoff=6, border_line_color=None, location=(0, 0))
            fig.add_layout(feature, 'right')
        if self.enable_notebook:
            self.show(fig, notebook_handle=True)
        return fig

    
    def __interactive_barchart(self, index, describer, tools='hover, pan, reset, save, wheel_zoom, zoom_in, zoom_out',
                                width=1000, height=400, toolbar_location='above',
                                sizing_mode='fixed', line_color=None, grid_line_color=None, axis_line_color=None,
                                major_tick_line_color=None, major_label_text_font_size='9pt',
                                major_label_standoff=0, title=True, bin_height=0.5,
                                transpose_data=False, bar_color='#053967'):
        """Plots an interactive barchart.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            bar_color (str), optional: Color of bars. Defaults to ``#053967``.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            tools (str), optional: Tools, which will be includeded. Defaults to ``hover,
                pan, reset, save, wheel_zoom, zoom_in, zoom_out``.
            width (int), optional: Width of the figure. Defaults to 1000.
            height (int), optional: Height of the figure. Defaults to 400.
            x_axis_location (str), optional: Location of the x-axis. Defaults to
                ``below``.
            toolbar_location (str), optional: Location of the toolbar. Defaults to
                ``above``.
            sizing_mode (str), optional: Size fixed or width oriented. Defaults to ``fixed``.
            line_color (str): Color for lines. Defaults to None.
            grid_line_color (str): Color for grid lines. Defaults to None.
            axis_line_color (str): Color for axis lines. Defaults to None.
            major_tick_line_color (str): Color for major tick lines. Defaults to None.
            major_label_text_font_size (str): Font size for major label text. Defaults
                to ``9pt``.
            major_label_standoff (int): Standoff for major labels. Defaults to 0.

        Returns:
            Figure object.
        """
        if isinstance(index, int):
            if transpose_data:
                proportions = self.document_topics.T.iloc[index]
            else:
                proportions = self.document_topics.iloc[index]
            if title:
                plot_title = '{}: {}'.format(describer, proportions.name)
        elif isinstance(index, str):
            if transpose_data:
                proportions = self.document_topics.T.loc[index]
            else:
                proportions = self.document_topics.loc[index]
            if title:
                plot_title = '{}: {}'.format(describer, index)
        else:
            raise ValueError("{} must be int or str.".format(index))

        x_axis = proportions
        y_range = list(proportions.index)

        source = ColumnDataSource(dict(Describer=y_range, Proportion=x_axis))

        fig = figure(y_range=y_range, title=plot_title, plot_width=width, plot_height=height,
                   tools=tools, toolbar_location=toolbar_location,
                   sizing_mode=sizing_mode, logo=None)
        fig.hbar(y='Describer', right='Proportion', height=bin_height, source=source,
               line_color=line_color, color=bar_color)

        fig.xgrid.grid_line_color = None
        fig.x_range.start = 0
        fig.grid.grid_line_color = grid_line_color
        fig.axis.axis_line_color = axis_line_color
        fig.axis.major_tick_line_color = major_tick_line_color
        fig.axis.major_label_text_font_size = major_label_text_font_size
        fig.axis.major_label_standoff = major_label_standoff
        
        if 'hover' in tools:
            fig.select_one(HoverTool).tooltips = [('Proportion', '@Proportion')]
        if self.enable_notebook:
            self.show(fig, notebook_handle=True)
        return fig    
    
    def interactive_barchart_per_topic(self, **kwargs):
        """Plots an interactive barchart per topic.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            bar_color (str), optional: Color of bars. Defaults to ``#053967``.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            tools (str), optional: Tools, which will be includeded. Defaults to ``hover,
                pan, reset, save, wheel_zoom, zoom_in, zoom_out``.
            width (int), optional: Width of the figure. Defaults to 1000.
            height (int), optional: Height of the figure. Defaults to 400.
            x_axis_location (str), optional: Location of the x-axis. Defaults to
                ``below``.
            toolbar_location (str), optional: Location of the toolbar. Defaults to
                ``above``.
            sizing_mode (str), optional: Size fixed or width oriented. Defaults to ``fixed``.
            line_color (str): Color for lines. Defaults to None.
            grid_line_color (str): Color for grid lines. Defaults to None.
            axis_line_color (str): Color for axis lines. Defaults to None.
            major_tick_line_color (str): Color for major tick lines. Defaults to None.
            major_label_text_font_size (str): Font size for major label text. Defaults
                to ``9pt``.
            major_label_standoff (int): Standoff for major labels. Defaults to 0.

        Returns:
            Figure object.
        """
        return self.__interactive_barchart(**kwargs)

    def interactive_barchart_per_document(self, **kwargs):
        """Plots an interactive barchart per document.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            bar_color (str), optional: Color of bars. Defaults to ``#053967``.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            tools (str), optional: Tools, which will be includeded. Defaults to ``hover,
                pan, reset, save, wheel_zoom, zoom_in, zoom_out``.
            width (int), optional: Width of the figure. Defaults to 1000.
            height (int), optional: Height of the figure. Defaults to 400.
            x_axis_location (str), optional: Location of the x-axis. Defaults to
                ``below``.
            toolbar_location (str), optional: Location of the toolbar. Defaults to
                ``above``.
            sizing_mode (str), optional: Size fixed or width oriented. Defaults to ``fixed``.
            line_color (str): Color for lines. Defaults to None.
            grid_line_color (str): Color for grid lines. Defaults to None.
            axis_line_color (str): Color for axis lines. Defaults to None.
            major_tick_line_color (str): Color for major tick lines. Defaults to None.
            major_label_text_font_size (str): Font size for major label text. Defaults
                to ``9pt``.
            major_label_standoff (int): Standoff for major labels. Defaults to 0.

        Returns:
            Figure object.
        """
        return self.__interactive_barchart(**kwargs, transpose_data=True)

    def topic_over_time(self, pattern = r"\d{4}",  threshold=0.1, starttime=1841, endtime=1920):
        """Creates a visualization that shows topics over time.

        Description:
            With this function you can plot topics over time using metadata stored in the documents name.
            Only works with mallet output.

        Args:
            labels(list): first three keys in a topic to select
            threshold(float): threshold set to define if a topic in a document is viable
            starttime(int): sets starting point for visualization
            endtime(int): sets ending point for visualization


        Returns: 
            matplotlib plot

        Note: this function is created for a corpus with filenames that looks like:
                1866_ArticleName.txt

        ToDo: make it compatible with gensim output
                Doctest

        """
        years=list(range(starttime,endtime))
        #doc_topicT = doc_topics.T
        topiclabels = []
        for topiclabel in doc_topics.index.values:
            for topiclabel in topiclabels:
                topic_over_threshold_per_year = []
                mask = doc_topics.loc[topiclabel] > threshold
                df = doc_topics.loc[topiclabel].loc[mask]
                #df = doc_topics.loc[doc_topics.loc[topiclabel] >  threshold]
                #print (df)
                d = defaultdict(int)
                for item in df.index.values:
                    reg = regex.compile(pattern)
                    year = reg.findall(item)
                    d[year[0]]+=1
                for year in years:
                    topic_over_threshold_per_year.append(d[str(year)])
                plt.plot(years, topic_over_threshold_per_year, label=topiclabel)

        plt.xlabel('Year')
        plt.ylabel('count topics over threshold')
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        return fig

    @staticmethod
    def to_file(fig, filename):
        """Saves a figure object to file.
    
        Args:
            fig Union(bokeh.figure, matplotlib.figure): Figure produced by either
                bokeh or matplotlib.
            filename (str): Name of the file with an extension, e.g. ``plot.png``.

        Returns:
            None.
        """
        import matplotlib
        import bokeh
        if isinstance(fig, bokeh.plotting.figure.Figure):
            ext = os.path.splitext(filename)[1]
            if ext == '.png':
                export_png(fig, filename)
            elif ext == '.svg':
                fig.output_backend = 'svg'
                export_svgs(fig, filename)
            elif ext == '.html':
                output_file(filename)
        elif isinstance(fig, matplotlib.figure.Figure):
            fig.savefig(filename)
        return None



