import streamlit as st
import pandas as pd
import altair as alt
from altair.utils.data import MaxRowsError
import numpy as np
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime


def setup_altair():
    """Configure Altair for handling large datasets with VegaFusion"""
    try:
        import vegafusion as vf
        # Enable vegafusion transformer
        alt.data_transformers.enable('vegafusion')
        # Optionally set timezone if needed
        # vf.set_local_tz("UTC")
    except ImportError:
        st.warning("VegaFusion not available, falling back to default transformer with no row limit")
        alt.data_transformers.disable_max_rows()



def load_custom_fonts():
    """Load custom fonts from the fonts directory"""
    font_dir = "fonts"
    if os.path.exists(font_dir):
        custom_fonts = []
        # Custom fonts dictionary
        custom_fonts_dict = {
            "BNazanin": "BNazanin.ttf",
            "IRANYekanXFaNum": "IRANYekanXFaNum-Medium.ttf",
            "Inter": "Inter_24pt-Medium.ttf",
            "Pangram Sans": "PPPangramSansRounded-Medium.otf"
        }
        
        for font_name, font_file in custom_fonts_dict.items():
            font_path = os.path.join(font_dir, font_file)
            if os.path.exists(font_path):
                try:
                    fm.fontManager.addfont(font_path)
                    font = fm.FontProperties(fname=font_path)
                    custom_fonts.append(font.get_name())
                except Exception as e:
                    st.warning(f"Failed to load font {font_file}: {str(e)}")
        
        return custom_fonts
    return []

def init_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'system_fonts' not in st.session_state:
        system_fonts = sorted(list(set([f.name for f in fm.fontManager.ttflist])))
        custom_fonts = load_custom_fonts()
        all_fonts = sorted(list(set(system_fonts + custom_fonts)))
        st.session_state.system_fonts = all_fonts
    if 'chart_type' not in st.session_state:
        st.session_state.chart_type = 'bar'
    if 'current_chart' not in st.session_state:
        st.session_state.current_chart = None

def process_data(df, settings):
    """Process data based on aggregation settings"""
    if df is None:
        return None
    
    try:
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Apply Top N / Bottom N filter
        if settings.get('filter_type') == 'Top N':
            processed_df = processed_df.nlargest(settings['filter_n'], settings['value_column'])
        elif settings.get('filter_type') == 'Bottom N':
            processed_df = processed_df.nsmallest(settings['filter_n'], settings['value_column'])
        
        # Apply aggregation
        if settings.get('aggregation') != 'None':
            if settings['aggregation'] == 'Sum':
                processed_df = processed_df.groupby(settings['category_column'])[settings['value_column']].sum().reset_index()
            elif settings['aggregation'] == 'Average':
                processed_df = processed_df.groupby(settings['category_column'])[settings['value_column']].mean().reset_index()
            elif settings['aggregation'] == 'Count':
                processed_df = processed_df.groupby(settings['category_column']).size().reset_index(name=settings['value_column'])
            elif settings['aggregation'] == 'Count (Non-Empty)':
                processed_df = processed_df.groupby(settings['category_column'])[settings['value_column']].count().reset_index()
        
        return processed_df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def create_bar_chart(df, settings):
    """Create bar chart with improved styling"""
    try:
        # Process data
        df = process_data(df, settings)
        if df is None:
            return None
        
        # Sort data
        df_sorted = df.sort_values(by=settings['value_column'], ascending=False)
        
        # Configure axes based on orientation
        if settings['orientation'] == "vertical":
            x_axis = alt.X(
                f'{settings["category_column"]}:N',
                title=settings['category_column'],
                sort=df_sorted[settings['category_column']].tolist(),
                axis=alt.Axis(
                    labelFont=settings['font_family'],
                    labelFontSize=settings['font_size'],
                    titleFont=settings['font_family'],
                    titleFontSize=settings['font_size'],
                    labelAngle=settings['label_angle']
                )
            )
            y_axis = alt.Y(
                f'{settings["value_column"]}:Q',
                title=settings['value_column'],
                axis=alt.Axis(
                    labelFont=settings['font_family'],
                    labelFontSize=settings['font_size'],
                    titleFont=settings['font_family'],
                    titleFontSize=settings['font_size']
                )
            )
        else:
            y_axis = alt.Y(
                f'{settings["category_column"]}:N',
                title=settings['category_column'],
                sort=df_sorted[settings['category_column']].tolist(),
                axis=alt.Axis(
                    labelFont=settings['font_family'],
                    labelFontSize=settings['font_size'],
                    titleFont=settings['font_family'],
                    titleFontSize=settings['font_size'],
                    labelAngle=settings['label_angle']
                )
            )
            x_axis = alt.X(
                f'{settings["value_column"]}:Q',
                title=settings['value_column'],
                axis=alt.Axis(
                    labelFont=settings['font_family'],
                    labelFontSize=settings['font_size'],
                    titleFont=settings['font_family'],
                    titleFontSize=settings['font_size']
                )
            )

        # Create base chart
        base = alt.Chart(df_sorted).mark_bar(
            cornerRadiusTopLeft=settings['corner_radius'],
            cornerRadiusTopRight=settings['corner_radius'],
            cornerRadiusBottomLeft=settings['corner_radius'],
            cornerRadiusBottomRight=settings['corner_radius']
        ).encode(
            x=x_axis,
            y=y_axis
        )

        # Apply color settings
        if settings['color_type'] == 'Solid':
            chart = base.encode(color=alt.value(settings['solid_color']))
        else:
            # Add normalized position for gradient
            df_sorted['normalized_value'] = (df_sorted[settings['value_column']] - 
                                           df_sorted[settings['value_column']].min()) / \
                                          (df_sorted[settings['value_column']].max() - 
                                           df_sorted[settings['value_column']].min())
            
            chart = alt.Chart(df_sorted).mark_bar(
                cornerRadiusTopLeft=settings['corner_radius'],
                cornerRadiusTopRight=settings['corner_radius'],
                cornerRadiusBottomLeft=settings['corner_radius'],
                cornerRadiusBottomRight=settings['corner_radius']
            ).encode(
                x=x_axis,
                y=y_axis,
                color=alt.Color(
                    'normalized_value:Q',
                    scale=alt.Scale(range=[settings['gradient_start'], settings['gradient_end']]),
                    legend=None
                )
            )

        # Add value labels if enabled
        if settings['show_labels']:
            if settings['orientation'] == "vertical":
                text = base.mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5,
                    fontSize=settings['font_size'],
                    font=settings['font_family']
                ).encode(
                    text=alt.Text(f'{settings["value_column"]}:Q', format='.0f')
                )
            else:
                text = base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=5,
                    fontSize=settings['font_size'],
                    font=settings['font_family']
                ).encode(
                    text=alt.Text(f'{settings["value_column"]}:Q', format='.0f')
                )
            chart = alt.layer(chart, text)

        return chart.properties(
            width=800,
            height=500
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            grid=False
        )
    
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")
        return None

def create_pie_chart(df, settings):
    """Create pie chart with improved styling"""
    try:
        # Process data
        df = process_data(df, settings)
        if df is None:
            return None
        
        fig = go.Figure(data=[go.Pie(
            labels=df[settings['category_column']],
            values=df[settings['value_column']],
            textfont=dict(
                family=settings['font_family'],
                size=settings['font_size']
            ),
            marker=dict(
                colors=getattr(px.colors.qualitative, settings['color_scheme'])
            ),
            textinfo='percent+label' if settings['show_labels'] else 'percent'
        )])
        
        fig.update_layout(
            font=dict(
                family=settings['font_family'],
                size=settings['font_size']
            ),
            showlegend=settings['show_legend']
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating pie chart: {str(e)}")
        return None

def create_donut_chart(df, settings):
    """Create donut chart with customizable settings"""
    try:
        # Process data
        df = process_data(df, settings)
        if df is None:
            return None
        
        fig = go.Figure(data=[go.Pie(
            labels=df[settings['category_column']],
            values=df[settings['value_column']],
            hole=settings['hole_size'],
            textfont=dict(
                family=settings['font_family'],
                size=settings['font_size']
            ),
            marker=dict(
                colors=getattr(px.colors.qualitative, settings['color_scheme']),
                line=dict(color='white', width=settings['segment_spacing'])
            ),
            textinfo='percent+label' if settings['show_labels'] else 'percent'
        )])
        
        fig.update_layout(
            font=dict(
                family=settings['font_family'],
                size=settings['font_size']
            ),
            showlegend=settings['show_legend']
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating donut chart: {str(e)}")
        return None

def create_line_chart(df, settings):
    """Create line chart with improved styling"""
    try:
        # Process data
        df = process_data(df, settings)
        if df is None:
            return None
        
        if settings['sort_data']:
            df = df.sort_values(by=[settings['category_column']])
        
        chart = alt.Chart(df).mark_line(
            point=settings['show_points']
        ).encode(
            x=alt.X(
                f'{settings["category_column"]}:N',
                title=settings['category_column'],
                axis=alt.Axis(
                    labelFont=settings['font_family'],
                    labelFontSize=settings['font_size'],
                    titleFont=settings['font_family'],
                    titleFontSize=settings['font_size']
                )
            ),
            y=alt.Y(
                f'{settings["value_column"]}:Q',
                title=settings['value_column'],
                axis=alt.Axis(
                    labelFont=settings['font_family'],
                    labelFontSize=settings['font_size'],
                    titleFont=settings['font_family'],
                    titleFontSize=settings['font_size']
                )
            ),
            color=alt.value(settings['line_color'])
        )

        if settings['show_labels']:
            text = chart.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                fontSize=settings['font_size'],
                font=settings['font_family']
            ).encode(
                text=alt.Text(f'{settings["value_column"]}:Q', format='.0f')
            )
            chart = alt.layer(chart, text)

        return chart.properties(
            width=800,
            height=500
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            grid=False
        )
    
    except Exception as e:
        st.error(f"Error creating line chart: {str(e)}")
        return None

def create_area_chart(df, settings):
    """Create area chart with smooth line option"""
    try:
        # Process data
        df = process_data(df, settings)
        if df is None:
            return None
        
        if settings['sort_data']:
            df = df.sort_values(by=[settings['category_column']])
        
        chart = alt.Chart(df).mark_area(
            opacity=settings['area_opacity'],
            interpolate=settings['interpolation']
        ).encode(
            x=alt.X(
                f'{settings["category_column"]}:N',
                title=settings['category_column'],
                axis=alt.Axis(
                    labelFont=settings['font_family'],
                    labelFontSize=settings['font_size'],
                    titleFont=settings['font_family'],
                    titleFontSize=settings['font_size']
                )
            ),
            y=alt.Y(
                f'{settings["value_column"]}:Q',
                title=settings['value_column'],
                axis=alt.Axis(
                    labelFont=settings['font_family'],
                    labelFontSize=settings['font_size'],
                    titleFont=settings['font_family'],
                    titleFontSize=settings['font_size']
                )
            ),
            color=alt.value(settings['area_color'])
        )

        if settings['show_line']:
            line = alt.Chart(df).mark_line(
                color=settings['line_color'],
                strokeWidth=settings['line_width']
            ).encode(
                x=f'{settings["category_column"]}:N',
                y=f'{settings["value_column"]}:Q'
            )
            chart = alt.layer(chart, line)

        return chart.properties(
            width=800,
            height=500
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            grid=False
        )
    
    except Exception as e:
        st.error(f"Error creating area chart: {str(e)}")
        return None

def create_radar_chart(df, settings):
    """Create radar chart"""
    try:
        # Process data
        df = process_data(df, settings)
        if df is None:
            return None
        
        categories = df[settings['category_column']].tolist()
        values = df[settings['value_column']].tolist()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Complete the circle
            theta=categories + [categories[0]],  # Complete
            fill='toself',
            fillcolor=settings['fill_color'],
            opacity=settings['fill_opacity'],
            line=dict(
                color=settings['line_color'],
                width=settings['line_width']
            )
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1]
                )
            ),
            showlegend=False,
            font=dict(
                family=settings['font_family'],
                size=settings['font_size']
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
        return None

def create_scatter_plot(df, settings):
    """Create scatter plot with customization options"""
    try:
        fig = px.scatter(
            df,
            x=settings['x_column'],
            y=settings['y_column'],
            color=settings['color_column'] if settings['use_color'] else None,
            size=settings['size_column'] if settings['use_size'] else None,
            hover_data=[settings['hover_column']] if settings['show_hover'] else None,
            color_continuous_scale=settings['color_scale'] if settings['use_color'] else None,
            opacity=settings['point_opacity']
        )
        
        fig.update_traces(
            marker=dict(
                size=settings['marker_size'] if not settings['use_size'] else None,
                line=dict(width=settings['marker_line_width'])
            )
        )
        
        fig.update_layout(
            font=dict(
                family=settings['font_family'],
                size=settings['font_size']
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating scatter plot: {str(e)}")
        return None

def create_histogram(df, settings):
    """Create histogram with customization options"""
    try:
        fig = go.Figure(data=[go.Histogram(
            x=df[settings['value_column']],
            nbinsx=settings['num_bins'],
            opacity=settings['bar_opacity'],
            marker_color=settings['bar_color'],
            histnorm=settings['normalization']
        )])
        
        fig.update_layout(
            bargap=settings['bar_gap'],
            font=dict(
                family=settings['font_family'],
                size=settings['font_size']
            ),
            xaxis_title=settings['value_column'],
            yaxis_title='Count' if settings['normalization'] == '' else 'Frequency'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating histogram: {str(e)}")
        return None

def create_box_plot(df, settings):
    """Create box plot with customization options"""
    try:
        fig = go.Figure()
        
        for category in df[settings['category_column']].unique():
            values = df[df[settings['category_column']] == category][settings['value_column']]
            
            fig.add_trace(go.Box(
                y=values,
                name=str(category),
                boxpoints=settings['point_display'],
                marker_color=settings['box_color'],
                line_color=settings['line_color'],
                boxmean=settings['show_mean']
            ))
        
        fig.update_layout(
            font=dict(
                family=settings['font_family'],
                size=settings['font_size']
            ),
            showlegend=settings['show_legend']
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating box plot: {str(e)}")
        return None

def create_treemap(df, settings):
    """Create treemap with customization options"""
    try:
        fig = px.treemap(
            df,
            path=[settings['hierarchy_column']],
            values=settings['value_column'],
            color=settings['color_column'] if settings['use_color'] else None,
            color_continuous_scale=settings['color_scale'] if settings['use_color'] else None
        )
        
        fig.update_traces(
            marker=dict(
                line=dict(
                    width=settings['border_width'],
                    color=settings['border_color']
                )
            ),
            textfont=dict(
                family=settings['font_family'],
                size=settings['font_size']
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating treemap: {str(e)}")
        return None

def get_common_settings(columns):
    """Get common chart settings"""
    settings = {}
    
    st.sidebar.subheader("Data Settings")
    settings['category_column'] = st.sidebar.selectbox("Category Column", columns)
    settings['value_column'] = st.sidebar.selectbox("Value Column", columns)
    
    settings['aggregation'] = st.sidebar.selectbox(
        "Aggregation",
        ['None', 'Sum', 'Average', 'Count', 'Count (Non-Empty)']
    )
    
    # Filter settings
    settings['filter_type'] = st.sidebar.selectbox(
        "Filter Type",
        ['None', 'Top N', 'Bottom N']
    )
    
    if settings['filter_type'] != 'None':
        max_n = len(st.session_state.df)
        settings['filter_n'] = st.sidebar.number_input(
            f"Number of {settings['filter_type']} items",
            min_value=1,
            max_value=max_n,
            value=min(5, max_n)
        )
    
    # Appearance settings
    st.sidebar.subheader("Appearance Settings")
    settings['font_family'] = st.sidebar.selectbox("Font", st.session_state.system_fonts)
    settings['font_size'] = st.sidebar.number_input("Font Size", 8, 24, 16)
    settings['show_labels'] = st.sidebar.checkbox("Show Labels", True)
    
    return settings

def get_bar_settings():
    """Get bar chart specific settings"""
    settings = {}
    
    st.sidebar.subheader("Bar Chart Settings")
    settings['orientation'] = st.sidebar.radio("Orientation", ["vertical", "horizontal"])
    settings['label_angle'] = st.sidebar.select_slider(
        "Category Label Angle",
        options=[0, 45, 90, -45, -90],
        value=0
    )
    
    settings['color_type'] = st.sidebar.radio(
        "Color Type",
        ["Solid", "Whole Gradient"]
    )
    
    if settings['color_type'] == 'Solid':
        settings['solid_color'] = st.sidebar.color_picker("Bar Color", "#1f77b4")
    else:
        settings['gradient_start'] = st.sidebar.color_picker("Start Color", "#1f77b4")
        settings['gradient_end'] = st.sidebar.color_picker("End Color", "#7fdbff")
    
    settings['corner_radius'] = st.sidebar.number_input("Corner Radius", 0, 100, 15)
    
    return settings

def get_pie_settings():
    """Get pie chart specific settings"""
    settings = {}
    
    st.sidebar.subheader("Pie Chart Settings")
    settings['show_legend'] = st.sidebar.checkbox("Show Legend", True)
    settings['color_scheme'] = st.sidebar.selectbox(
        "Color Scheme",
        ['Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2']
    )
    
    return settings

def get_donut_settings():
    """Get donut chart specific settings"""
    settings = {}
    
    st.sidebar.subheader("Donut Chart Settings")
    settings['hole_size'] = st.sidebar.slider("Hole Size", 0.0, 0.9, 0.5)
    settings['segment_spacing'] = st.sidebar.slider("Segment Spacing", 0, 5, 2)
    settings['show_legend'] = st.sidebar.checkbox("Show Legend", True)
    settings['color_scheme'] = st.sidebar.selectbox(
        "Color Scheme",
        ['Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2']
    )
    
    return settings

def get_line_settings():
    """Get line chart specific settings"""
    settings = {}
    
    st.sidebar.subheader("Line Chart Settings")
    settings['line_color'] = st.sidebar.color_picker("Line Color", "#1f77b4")
    settings['show_points'] = st.sidebar.checkbox("Show Points", True)
    settings['sort_data'] = st.sidebar.checkbox("Sort X-Axis", False)
    
    return settings

def get_area_settings():
    """Get area chart specific settings"""
    settings = {}
    
    st.sidebar.subheader("Area Chart Settings")
    settings['area_opacity'] = st.sidebar.slider("Area Opacity", 0.0, 1.0, 0.5)
    settings['area_color'] = st.sidebar.color_picker("Area Color", "#1f77b4")
    settings['show_line'] = st.sidebar.checkbox("Show Line", True)
    settings['line_color'] = st.sidebar.color_picker("Line Color", "#000000")
    settings['line_width'] = st.sidebar.slider("Line Width", 1, 5, 2)
    settings['interpolation'] = st.sidebar.selectbox(
        "Line Style",
        ['linear', 'basis', 'cardinal', 'step']
    )
    settings['sort_data'] = st.sidebar.checkbox("Sort X-Axis", False)
    
    return settings

def get_radar_settings():
    """Get radar chart specific settings"""
    settings = {}
    
    st.sidebar.subheader("Radar Chart Settings")
    settings['fill_color'] = st.sidebar.color_picker("Fill Color", "#1f77b4")
    settings['fill_opacity'] = st.sidebar.slider("Fill Opacity", 0.0, 1.0, 0.5)
    settings['line_color'] = st.sidebar.color_picker("Line Color", "#000000")
    settings['line_width'] = st.sidebar.slider("Line Width", 1, 5, 2)
    
    return settings

def get_scatter_settings(columns):
    """Get scatter plot specific settings"""
    settings = {}
    
    st.sidebar.subheader("Scatter Plot Settings")
    settings['x_column'] = st.sidebar.selectbox("X-Axis Column", columns)
    settings['y_column'] = st.sidebar.selectbox("Y-Axis Column", columns)
    
    settings['use_color'] = st.sidebar.checkbox("Use Color Mapping", False)
    if settings['use_color']:
        settings['color_column'] = st.sidebar.selectbox("Color Column", columns)
        settings['color_scale'] = st.sidebar.selectbox(
            "Color Scale",
            ['Viridis', 'Plasma', 'Inferno', 'Magma']
        )
    
    settings['use_size'] = st.sidebar.checkbox("Use Size Mapping", False)
    if settings['use_size']:
        settings['size_column'] = st.sidebar.selectbox("Size Column", columns)
    else:
        settings['marker_size'] = st.sidebar.slider("Marker Size", 5, 30, 10)
    
    settings['marker_line_width'] = st.sidebar.slider("Marker Line Width", 0, 5, 1)
    settings['point_opacity'] = st.sidebar.slider("Point Opacity", 0.0, 1.0, 0.7)
    
    settings['show_hover'] = st.sidebar.checkbox("Show Hover Data", True)
    if settings['show_hover']:
        settings['hover_column'] = st.sidebar.selectbox("Hover Data Column", columns)
    
    return settings

def get_histogram_settings():
    """Get histogram specific settings"""
    settings = {}
    
    st.sidebar.subheader("Histogram Settings")
    settings['num_bins'] = st.sidebar.slider("Number of Bins", 5, 100, 30)
    settings['bar_opacity'] = st.sidebar.slider("Bar Opacity", 0.0, 1.0, 0.7)
    settings['bar_color'] = st.sidebar.color_picker("Bar Color", "#1f77b4")
    settings['bar_gap'] = st.sidebar.slider("Bar Gap", 0.0, 0.5, 0.1)
    settings['normalization'] = st.sidebar.selectbox(
        "Normalization",
        ['', 'percent', 'probability', 'density']
    )
    
    return settings

def get_box_settings():
    """Get box plot specific settings"""
    settings = {}
    
    st.sidebar.subheader("Box Plot Settings")
    settings['point_display'] = st.sidebar.selectbox(
        "Show Points",
        ['all', 'outliers', 'suspected outliers', False]
    )
    settings['box_color'] = st.sidebar.color_picker("Box Color", "#1f77b4")
    settings['line_color'] = st.sidebar.color_picker("Line Color", "#000000")
    settings['show_mean'] = st.sidebar.checkbox("Show Mean", True)
    settings['show_legend'] = st.sidebar.checkbox("Show Legend", True)
    
    return settings

def get_treemap_settings(columns):
    """Get treemap specific settings"""
    settings = {}
    
    st.sidebar.subheader("Treemap Settings")
    settings['hierarchy_column'] = st.sidebar.selectbox("Hierarchy Column", columns)
    
    settings['use_color'] = st.sidebar.checkbox("Use Color Mapping", False)
    if settings['use_color']:
        settings['color_column'] = st.sidebar.selectbox("Color Column", columns)
        settings['color_scale'] = st.sidebar.selectbox(
            "Color Scale",
            ['Viridis', 'Plasma', 'Inferno', 'Magma']
        )
    
    settings['border_width'] = st.sidebar.slider("Border Width", 0, 5, 1)
    settings['border_color'] = st.sidebar.color_picker("Border Color", "#FFFFFF")
    
    return settings

def get_chart_settings(chart_type):
    """Get chart settings based on chart type"""
    settings = {}
    
    if st.session_state.df is not None:
        columns = list(st.session_state.df.columns)
        
        # Get common settings
        settings.update(get_common_settings(columns))
        
        # Get chart-specific settings
        if chart_type == 'bar':
            settings.update(get_bar_settings())
        elif chart_type == 'pie':
            settings.update(get_pie_settings())
        elif chart_type == 'donut':
            settings.update(get_donut_settings())
        elif chart_type == 'line':
            settings.update(get_line_settings())
        elif chart_type == 'area':
            settings.update(get_area_settings())
        elif chart_type == 'radar':
            settings.update(get_radar_settings())
        elif chart_type == 'scatter':
            settings.update(get_scatter_settings(columns))
        elif chart_type == 'histogram':
            settings.update(get_histogram_settings())
        elif chart_type == 'box':
            settings.update(get_box_settings())
        elif chart_type == 'treemap':
            settings.update(get_treemap_settings(columns))
        return settings

def save_chart(chart, format_type, ppi=None, scale_factor=None):
    """Save chart using Altair's native save methods"""
    try:
        import io
        
        if isinstance(chart, (alt.Chart, alt.LayerChart)):
            try:
                if format_type == 'html':
                    # Use Altair's html saving
                    buffer = io.StringIO()
                    chart.save(buffer, format='html')
                    return buffer.getvalue().encode('utf-8')
                
                elif format_type == 'svg':
                    # Use Altair's svg saving
                    buffer = io.StringIO()
                    chart.save(buffer, format='svg')
                    return buffer.getvalue().encode('utf-8')
                
                else:  # png
                    # Use Altair's png saving
                    buffer = io.BytesIO()
                    
                    # Apply scaling if needed
                    if scale_factor is not None or ppi is not None:
                        scale = scale_factor if scale_factor is not None else ppi/72
                        chart = chart.properties(
                            width=chart.width * scale,
                            height=chart.height * scale
                        )
                    
                    chart.save(buffer, format='png')
                    buffer.seek(0)
                    return buffer.getvalue()
                    
            except Exception as save_error:
                st.error(f"Save error: {str(save_error)}")
                return None
                    
        else:  # Plotly chart
            if format_type == 'html':
                buffer = io.StringIO()
                chart.write_html(buffer)
                buffer.seek(0)
                return buffer.getvalue().encode('utf-8')
            else:  # svg or png
                buffer = io.BytesIO()
                
                # Update layout to ensure white background
                chart.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                if format_type == 'svg':
                    chart.write_image(buffer, format='svg')
                else:  # png
                    if scale_factor is not None:
                        width = int(800 * scale_factor)
                        height = int(500 * scale_factor)
                        chart.write_image(buffer, format='png', width=width, height=height)
                    else:
                        chart.write_image(buffer, format='png')
                buffer.seek(0)
                return buffer.getvalue()
            
    except Exception as e:
        st.error(f"Error preparing chart for download: {str(e)}")
        return None
   

def main():
    setup_altair()
    
    st.set_page_config(page_title="Chart Creator", layout="wide")
    init_session_state()

    
    # Create two main columns for layout
    left_col, right_col = st.columns([1, 3])
    
    # Left column - Controls
    with left_col:
        st.title("Chart Creator")
        
        # File Upload
        uploaded_file = st.file_uploader("Upload Data File", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success("File loaded successfully!")
                with st.expander("Data Preview"):
                    st.dataframe(st.session_state.df.head())
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        # Chart Type Selection
        chart_types = [
            'bar', 'pie', 'donut', 'line', 'area', 'radar',
            'scatter', 'histogram', 'box', 'treemap'
        ]
        chart_type = st.selectbox(
            "Select Chart Type",
            chart_types,
            index=chart_types.index(st.session_state.chart_type)
        )
        st.session_state.chart_type = chart_type
        
        # Save options
        if st.session_state.df is not None and st.session_state.current_chart is not None:
            st.write("### Export Options")
            format_type = st.selectbox(
                "Select Format",
                ['svg', 'html', 'png']
            )
            
            # PNG options
            ppi = None
            scale_factor = None
            if format_type == 'png':
                col1, col2 = st.columns(2)
                with col1:
                    ppi = st.number_input("Resolution (PPI)", 
                                        min_value=72, 
                                        max_value=300, 
                                        value=72,
                                        help="Pixels per inch (72 is default)")
                with col2:
                    scale_factor = st.number_input("Scale Factor",
                                                min_value=0.1,
                                                max_value=5.0,
                                                value=1.0,
                                                step=0.1,
                                                help="Size multiplier (1.0 is default)")
            
            # Get chart data
            chart_data = save_chart(
                st.session_state.current_chart,
                format_type,
                ppi=ppi,
                scale_factor=scale_factor
            )
            
            if chart_data is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mime_type = {
                    'html': 'text/html',
                    'svg': 'image/svg+xml',
                    'png': 'image/png'
                }[format_type]
                
                st.download_button(
                    label="Download Chart",
                    data=chart_data,
                    file_name=f"chart_{timestamp}.{format_type}",
                    mime=mime_type,
                    use_container_width=True
                )


    # Right column - Chart Preview
    with right_col:
        if st.session_state.df is not None:
            try:
                # Get settings
                settings = get_chart_settings(chart_type)
                
                # Create appropriate chart
                if chart_type == 'bar':
                    chart = create_bar_chart(st.session_state.df, settings)
                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                elif chart_type == 'pie':
                    chart = create_pie_chart(st.session_state.df, settings)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                elif chart_type == 'donut':
                    chart = create_donut_chart(st.session_state.df, settings)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                elif chart_type == 'line':
                    chart = create_line_chart(st.session_state.df, settings)
                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                elif chart_type == 'area':
                    chart = create_area_chart(st.session_state.df, settings)
                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                elif chart_type == 'radar':
                    chart = create_radar_chart(st.session_state.df, settings)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                elif chart_type == 'scatter':
                    chart = create_scatter_plot(st.session_state.df, settings)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                elif chart_type == 'histogram':
                    chart = create_histogram(st.session_state.df, settings)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                elif chart_type == 'box':
                    chart = create_box_plot(st.session_state.df, settings)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                elif chart_type == 'treemap':
                    chart = create_treemap(st.session_state.df, settings)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                
                st.session_state.current_chart = chart
                
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
                with st.expander("Debug Information"):
                    st.code(str(e))
        else:
            st.info("Please upload a data file to begin")

if __name__ == "__main__":
    main()
