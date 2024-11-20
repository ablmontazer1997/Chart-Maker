import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime


def load_custom_fonts():
    """Load Persian and custom fonts from the fonts directory"""
    font_dir = "fonts"
    if os.path.exists(font_dir):
        custom_fonts = []
        # Persian fonts
        persian_fonts = {
            "BNazanin": "BNazanin.ttf",
            "IRANYekanXFaNum": "IRANYekanXFaNum-Medium.ttf",
            "Inter": "Inter_24pt-Medium.ttf"
        }
        
        for font_name, font_file in persian_fonts.items():
            font_path = os.path.join(font_dir, font_file)
            if os.path.exists(font_path):
                try:
                    # Add font to matplotlib
                    fm.fontManager.addfont(font_path)
                    # Get font family name
                    font = fm.FontProperties(fname=font_path)
                    custom_fonts.append(font.get_name())
                except Exception as e:
                    st.warning(f"Failed to load font {font_file}: {str(e)}")
                    
        # Add any additional fonts in the directory
        for font_file in os.listdir(font_dir):
            if font_file.endswith(('.ttf', '.otf')) and font_file not in persian_fonts.values():
                font_path = os.path.join(font_dir, font_file)
                try:
                    fm.fontManager.addfont(font_path)
                    font = fm.FontProperties(fname=font_path)
                    custom_fonts.append(font.get_name())
                except Exception as e:
                    st.warning(f"Failed to load font {font_file}: {str(e)}")
                    
        return custom_fonts
    return []


# Page config
st.set_page_config(page_title="Chart Creator (Abolfazl Montazer)", layout="wide")

def init_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'system_fonts' not in st.session_state:
        # Get system fonts
        system_fonts = sorted(list(set([f.name for f in fm.fontManager.ttflist])))
        # Add Persian fonts
        custom_fonts = load_custom_fonts()
        # Combine and sort all fonts
        all_fonts = sorted(list(set(system_fonts + custom_fonts)))
        st.session_state.system_fonts = all_fonts
    if 'chart_type' not in st.session_state:
        st.session_state.chart_type = 'bar'
    if 'current_chart' not in st.session_state:
        st.session_state.current_chart = None

# Add custom CSS to support custom fonts
def add_custom_css():
    """Add custom CSS for Persian fonts"""
    font_dir = "fonts"
    if os.path.exists(font_dir):
        css = """
        <style>
        @font-face {
            font-family: 'BNazanin';
            src: url('data:font/truetype;charset=utf-8;base64,%s') format('truetype');
        }
        @font-face {
            font-family: 'IRANYekanXFaNum';
            src: url('data:font/truetype;charset=utf-8;base64,%s') format('truetype');
        }
        @font-face {
            font-family: 'Inter';
            src: url('data:font/truetype;charset=utf-8;base64,%s') format('truetype');
        }
        .stMarkdown {
            font-family: 'IRANYekanXFaNum', 'BNazanin', 'Inter', sans-serif;
        }
        </style>
        """ % (
            get_font_base64(os.path.join(font_dir, "BNazanin.ttf")),
            get_font_base64(os.path.join(font_dir, "IRANYekanXFaNum-Medium.ttf")),
            get_font_base64(os.path.join(font_dir, "Inter_24pt-Medium.ttf"))
        )
        st.markdown(css, unsafe_allow_html=True)

def get_font_base64(font_path):
    """Convert font file to base64"""
    import base64
    try:
        with open(font_path, "rb") as font_file:
            return base64.b64encode(font_file.read()).decode()
    except Exception as e:
        st.warning(f"Failed to load font {font_path}: {str(e)}")
        return ""


def process_data(df, settings):
    """Process data based on aggregation settings"""
    if df is None:
        return None
        
    # Apply Top N / Bottom N filter
    if settings.get('filter_type') == 'Top N':
        df = df.nlargest(settings['filter_n'], settings['value_column'])
    elif settings.get('filter_type') == 'Bottom N':
        df = df.nsmallest(settings['filter_n'], settings['value_column'])
        
    # Apply aggregation
    if settings.get('aggregation') == 'Sum':
        df = df.groupby(settings['category_column'])[settings['value_column']].sum().reset_index()
    elif settings.get('aggregation') == 'Average':
        df = df.groupby(settings['category_column'])[settings['value_column']].mean().reset_index()
    elif settings.get('aggregation') == 'Count':
        df = df.groupby(settings['category_column']).size().reset_index(name=settings['value_column'])
    elif settings.get('aggregation') == 'Count (Non-Empty)':
        df = df.groupby(settings['category_column'])[settings['value_column']].count().reset_index()
        
    return df

def save_chart(chart, format_type, ppi=None, scale_factor=None):
    """Save chart using vl-convert-python"""
    try:
        import io
        import base64
        from datetime import datetime
        
        # Try importing vl-convert
        try:
            import vl_convert as vlc
        except ImportError:
            return False, """Error: Please install vl-convert-python for image export.
            Run: pip install vl-convert-python"""

        # Create buffer for the chart
        if isinstance(chart, (alt.Chart, alt.LayerChart)):
            if format_type == 'html':
                # Save as HTML with embedded dependencies
                content = chart.to_html(inline=True)
                mime_type = "text/html"
                data = content
            elif format_type in ['svg', 'png']:
                # Get the Vega-Lite spec
                spec = chart.to_dict()
                
                if format_type == 'svg':
                    # Convert to SVG
                    content = vlc.vegalite_to_svg(spec)
                    mime_type = "image/svg+xml"
                    data = content
                else:  # png
                    # Add PNG-specific options
                    vl_options = {}
                    if ppi is not None:
                        vl_options['ppi'] = ppi
                    if scale_factor is not None:
                        vl_options['scale_factor'] = scale_factor
                        
                    # Convert to PNG
                    content = vlc.vegalite_to_png(spec, **vl_options)
                    mime_type = "image/png"
                    data = base64.b64encode(content).decode()
        else:  # Plotly chart
            if format_type == 'html':
                buffer = io.StringIO()
                chart.write_html(buffer)
                buffer.seek(0)
                data = buffer.getvalue()
                mime_type = "text/html"
            else:  # svg or png
                buffer = io.BytesIO()
                if format_type == 'svg':
                    chart.write_image(buffer, format='svg')
                    mime_type = "image/svg+xml"
                else:
                    chart.write_image(buffer, format='png')
                    mime_type = "image/png"
                buffer.seek(0)
                data = base64.b64encode(buffer.getvalue()).decode()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{timestamp}.{format_type}"
        
        return True, (data, filename, mime_type)
            
    except Exception as e:
        return False, f"Error preparing chart for download: {str(e)}"
    

def create_bar_chart(df, settings):
    """Create bar chart with improved gradient and corner radius control"""
    # Process data first
    df = process_data(df, settings)
    
    # Sort data
    df_sorted = df.sort_values(by=settings['value_column'], ascending=False)
    
    # Configure axis label rotation
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
    else:  # horizontal
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

    # Create base chart with corner radius
    base = alt.Chart(df_sorted).mark_bar(
        cornerRadiusTopLeft=settings['corner_radius_top_left'],
        cornerRadiusTopRight=settings['corner_radius_top_right'],
        cornerRadiusBottomLeft=settings['corner_radius_bottom_left'],
        cornerRadiusBottomRight=settings['corner_radius_bottom_right']
    ).encode(
        x=x_axis,
        y=y_axis
    )

    # Apply color based on settings
    if settings['color_type'] == 'Solid':
        chart = base.encode(
            color=alt.value(settings['solid_color'])
        )
    elif settings['color_type'] == 'Individual Gradient':
        # Add normalized position for each category
        df_sorted['normalized_value'] = (df_sorted[settings['value_column']] - 
                                       df_sorted[settings['value_column']].min()) / \
                                      (df_sorted[settings['value_column']].max() - 
                                       df_sorted[settings['value_column']].min())
        
        chart = alt.Chart(df_sorted).transform_window(
            rank='rank()',
            sort=[alt.SortField(settings['value_column'], order='descending')]
        ).mark_bar(
            cornerRadiusTopLeft=settings['corner_radius_top_left'],
            cornerRadiusTopRight=settings['corner_radius_top_right'],
            cornerRadiusBottomLeft=settings['corner_radius_bottom_left'],
            cornerRadiusBottomRight=settings['corner_radius_bottom_right']
        ).encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color(
                'normalized_value:Q',
                scale=alt.Scale(range=[settings['gradient_start'], settings['gradient_end']]),
                legend=None
            )
        )
    else:  # Whole Gradient
        df_sorted['color_index'] = range(len(df_sorted))
        chart = base.encode(
            color=alt.Color(
                'color_index:Q',
                scale=alt.Scale(
                    domain=[0, len(df_sorted)-1],
                    range=[settings['gradient_start'], settings['gradient_end']]
                ),
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


def create_pie_chart(df, settings):
    """Create pie chart with updated settings"""
    # Process data first
    df = process_data(df, settings)
    
    # Create figure with explicit font settings
    fig = go.Figure(data=[go.Pie(
        labels=df[settings['category_column']],
        values=df[settings['value_column']],
        textfont=dict(
            family=settings['font_family'],
            size=settings['font_size']
        ),
        marker_colors=getattr(px.colors.qualitative, settings['color_scheme']),
        textinfo='percent+label' if settings['show_labels'] else 'percent'
    )])
    
    # Update layout with font settings
    fig.update_layout(
        font=dict(
            family=settings['font_family'],
            size=settings['font_size']
        ),
        showlegend=settings['show_legend']
    )
    
    return fig

def create_line_chart(df, settings):
    """Create line chart with settings"""
    # Process data first
    df = process_data(df, settings)
    
    # Sort if requested
    if settings['sort_data']:
        df = df.sort_values(by=[settings['category_column']])
    
    # Create chart
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

def create_sankey_chart(df, settings):
    """Create sankey diagram with settings"""
    # Process data first
    df = process_data(df, settings)
    
    # Get unique nodes
    source_nodes = df[settings['source_column']].unique()
    target_nodes = df[settings['target_column']].unique()
    all_nodes = list(set(np.concatenate([source_nodes, target_nodes])))
    
    # Create node dictionary
    node_dict = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color="blue"
        ),
        link=dict(
            source=[node_dict[src] for src in df[settings['source_column']]],
            target=[node_dict[tgt] for tgt in df[settings['target_column']]],
            value=df[settings['value_column']]
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text="Sankey Diagram",
        font=dict(
            family=settings['font_family'],
            size=settings['font_size']
        )
    )
    
    return fig

def get_chart_settings(chart_type):
    """Get chart settings with unified radius option"""
    settings = {}
    
    if st.session_state.df is not None:
        columns = list(st.session_state.df.columns)
        
        st.sidebar.subheader("Data Settings")
        settings['category_column'] = st.sidebar.selectbox("Category Column", columns)
        settings['value_column'] = st.sidebar.selectbox("Value Column", columns)
        
        # Data Processing Settings
        settings['aggregation'] = st.sidebar.selectbox(
            "Aggregation",
            ['None', 'Sum', 'Average', 'Count', 'Count (Non-Empty)']
        )
        
        # Calculate max_n based on actual data
        if settings['aggregation'] == 'None':
            max_n = len(st.session_state.df)
        else:
            grouped = st.session_state.df.groupby(settings['category_column'])
            max_n = len(grouped)
        
        settings['filter_type'] = st.sidebar.selectbox(
            "Filter Type",
            ['None', 'Top N', 'Bottom N']
        )
        
        if settings['filter_type'] != 'None':
            settings['filter_n'] = st.sidebar.number_input(
                f"Number of {settings['filter_type']} items",
                min_value=1,
                max_value=max_n,
                value=min(5, max_n)
            )
        
        st.sidebar.subheader("Appearance Settings")
        settings['font_family'] = st.sidebar.selectbox("Font", st.session_state.system_fonts)
        settings['font_size'] = st.sidebar.number_input("Font Size", 8, 24, 12)
        settings['show_labels'] = st.sidebar.checkbox("Show Labels", True)
        
        if chart_type == 'bar':
            st.sidebar.subheader("Bar Chart Settings")
            settings['orientation'] = st.sidebar.radio("Orientation", ["vertical", "horizontal"])
            
            # Label orientation
            settings['label_angle'] = st.sidebar.select_slider(
                "Category Label Angle",
                options=[0, 45, 90, -45, -90],
                value=0
            )
            
            # Color settings
            settings['color_type'] = st.sidebar.radio(
                "Color Type",
                ["Solid", "Individual Gradient", "Whole Gradient"]
            )
            
            if settings['color_type'] == 'Solid':
                settings['solid_color'] = st.sidebar.color_picker("Bar Color", "#1f77b4")
            else:
                settings['gradient_start'] = st.sidebar.color_picker("Start Color", "#1f77b4")
                settings['gradient_end'] = st.sidebar.color_picker("End Color", "#7fdbff")
            
            # Corner radius settings
            st.sidebar.subheader("Corner Radius")
            radius_type = st.sidebar.radio("Radius Type", ["Unified", "Individual"])
            
            if radius_type == "Unified":
                unified_radius = st.sidebar.number_input("Corner Radius", 0, 50, 0)
                settings['corner_radius_top_left'] = unified_radius
                settings['corner_radius_top_right'] = unified_radius
                settings['corner_radius_bottom_left'] = unified_radius
                settings['corner_radius_bottom_right'] = unified_radius
            else:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    settings['corner_radius_top_left'] = st.number_input("Top Left", 0, 50, 0)
                    settings['corner_radius_bottom_left'] = st.number_input("Bottom Left", 0, 50, 0)
                with col2:
                    settings['corner_radius_top_right'] = st.number_input("Top Right", 0, 50, 0)
                    settings['corner_radius_bottom_right'] = st.number_input("Bottom Right", 0, 50, 0)
            
        elif chart_type == 'pie':
            settings['show_legend'] = st.sidebar.checkbox("Show Legend", True)
            settings['color_scheme'] = st.sidebar.selectbox(
                "Color Scheme",
                ['Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2']
            )
            
        elif chart_type == 'line':
            settings['line_color'] = st.sidebar.color_picker("Line Color", "#1f77b4")
            settings['show_points'] = st.sidebar.checkbox("Show Points", True)
            settings['sort_data'] = st.sidebar.checkbox("Sort X-Axis", False)
            
        elif chart_type == 'sankey':
            settings['source_column'] = st.sidebar.selectbox("Source Column", columns)
            settings['target_column'] = st.sidebar.selectbox("Target Column", columns)
            settings['value_column'] = st.sidebar.selectbox("Value Column", columns)
    
    return settings


def main():
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
        chart_type = st.selectbox(
            "Select Chart Type",
            ['bar', 'pie', 'line', 'sankey'],
            index=['bar', 'pie', 'line', 'sankey'].index(st.session_state.chart_type)
        )
        st.session_state.chart_type = chart_type
        
        # Save options
        if st.session_state.df is not None:
            st.write("### Export Options")
            format_type = st.selectbox(
                "Select Format",
                ['html', 'svg', 'png']
            )
            
            # PNG options
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
            
            if st.session_state.current_chart:
                if st.button("Download Chart", use_container_width=True):
                    with st.spinner('Preparing download...'):
                        if format_type == 'png':
                            success, result = save_chart(
                                st.session_state.current_chart, 
                                format_type,
                                ppi=ppi,
                                scale_factor=scale_factor
                            )
                        else:
                            success, result = save_chart(
                                st.session_state.current_chart, 
                                format_type
                            )
                        
                        if success:
                            data, filename, mime_type = result
                            st.download_button(
                                label=f"Click to download {format_type.upper()}",
                                data=data,
                                file_name=filename,
                                mime=mime_type,
                                use_container_width=True
                            )
                        else:
                            st.error(result)

    
    # Right column - Chart Preview
    with right_col:
        if st.session_state.df is not None:
            try:
                # Get settings
                settings = get_chart_settings(chart_type)
                
                # Create appropriate chart
                if chart_type == 'bar':
                    chart = create_bar_chart(st.session_state.df, settings)
                    st.altair_chart(chart, use_container_width=True)
                elif chart_type == 'pie':
                    chart = create_pie_chart(st.session_state.df, settings)
                    st.plotly_chart(chart, use_container_width=True)
                elif chart_type == 'line':
                    chart = create_line_chart(st.session_state.df, settings)
                    st.altair_chart(chart, use_container_width=True)
                elif chart_type == 'sankey':
                    chart = create_sankey_chart(st.session_state.df, settings)
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
