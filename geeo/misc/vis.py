from ipywidgets import Output, Layout
from ipyleaflet import Map, TileLayer, DrawControl, LayersControl, WidgetControl, ScaleControl, GeoJSON, Popup
import ee
from ipywidgets import HTML
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from geeo.misc.spacetime import getRegion
import geopandas as gpd



# plot time series from getRegion
def plot_getRegion(imgcol, band, roi, scale=30, axis=None, style='.', color="k", label='', legend=True):
    
    df = getRegion(imgcol, roi, scale=scale)
    
    # Plotting
    if axis is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        ax = axis

    ax.plot(df['datetime'],
            df[band],
            style, color=color,
            label=label, lw=1.5)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.grid(lw=0.2)
    if legend:
        ax.legend(fontsize=11, loc='lower right')

    if axis is None:
        plt.show()

    return ax


def plot_rbf_interpolation(df, interp, value_col='NDVI', observed=False, ax=None, label='RBF', **kwargs):
    """
    Plot observed values and a single RBF interpolation.

    Args:
        df (pd.DataFrame): Original data with datetime index and value_col.
        interp (pd.DataFrame): RBF interpolation result.
        value_col (str): Name of the observed value column.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.
        label (str): Label for the interpolation line.
        **kwargs: Additional keyword arguments for the interpolation plot (e.g., color, linestyle).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
    if observed:
        ax.plot(df.index, df[value_col], 'kx', label='Observed')
    ax.plot(interp.index, interp['rbf_interp'], label=label, **kwargs)
    ax.set_title('RBF Interpolation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()
    if ax is None:
        plt.tight_layout()
        plt.show()
    return ax


# visualization using ipyleaflet
class VisMap:
    def __init__(self, zoom_start=2, width='100%', height='800px'):
        # Initialize the map
        self.map = Map(zoom=zoom_start, scroll_wheel_zoom=True, layout=Layout(width=width, height=height))
        self.out = Output()
        
        # Add a base layer (Google Satellite)
        google_layer = TileLayer(url='https://mt1.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}', 
                                 attribution='Google', name='Google Satellite')
        self.map.add_layer(google_layer)
        
        # Add controls
        self.draw_control = DrawControl(rectangle={"shapeOptions": {"color": "#ff7800", "weight": 1}})
        self.draw_control.on_draw(self.handle_draw)
        self.map.add_control(self.draw_control)
        
        self.layers_control = LayersControl(position='topright')
        self.map.add_control(self.layers_control)

        # Add the scale bar
        self.scale_control = ScaleControl(position='bottomleft')
        self.map.add_control(self.scale_control)
        
        output_control = WidgetControl(widget=self.out, position='bottomright')
        self.map.add_control(output_control)
    
    def create_color_ramp(self, colormap_name, n_colors=256):
        colormap = plt.get_cmap(colormap_name)
        colors = [mcolors.rgb2hex(colormap(i / n_colors)) for i in range(n_colors)]
        return colors

    def get_percentile_min_max(self, img, geometry, percentiles=[2, 98]):
        bands = img.bandNames().getInfo()
        percentile_values = img.reduceRegion(
            reducer=ee.Reducer.percentile(percentiles),
            geometry=geometry,
            scale=60,
            maxPixels=10000,
            bestEffort=True
        ).getInfo()

        min_vals = [percentile_values[f'{band}_p{percentiles[0]}'] for band in bands]
        max_vals = [percentile_values[f'{band}_p{percentiles[1]}'] for band in bands]
        
        return {
            'min': min_vals,
            'max': max_vals
        }

    def add_ee_layer(self, ee_object, vis_params=None, name='Image', opacity=1.0, cmap=None, roi=None):
        if vis_params is None:
            # If visualization parameters are not provided, calculate them using percentiles
            if roi is None:
                roi_geometry = ee_object.geometry()
            elif isinstance(roi, ee.Geometry):
                roi_geometry = roi
            else:
                roi_geometry = ee.Geometry.Rectangle(roi)
            vis_params = self.get_percentile_min_max(ee_object, roi_geometry)
        
        if cmap:
            # If a colormap is provided, apply it to the visualization parameters
            color_ramp = self.create_color_ramp(cmap)
            vis_params['palette'] = color_ramp
        
        map_id_dict = ee.Image(ee_object).getMapId(vis_params)
        tile_layer = TileLayer(
            url=map_id_dict['tile_fetcher'].url_format,
            attribution='Google Earth Engine',
            name=name,
            opacity=opacity
        )
        self.map.add_layer(tile_layer)

    def add_vector_layer(self, vector_data, layer_name='Vector Layer', color='blue', fill_color='blue', opacity=1.0, fill_opacity=0.5):
        """
        Adds a GeoPandas GeoDataFrame or a single geometry to the map, reprojecting to WGS84 if necessary.
        Makes the geometries clickable to show attribute values.

        Args:
            vector_data (GeoDataFrame or geometry): The vector data to visualize.
            layer_name (str): The name of the layer.
            color (str): The color of the lines.
            fill_color (str): The color of the fill for polygons.
            opacity (float): The opacity of the lines.
            fill_opacity (float): The opacity of the fill for polygons.
        """
        if isinstance(vector_data, gpd.GeoDataFrame):
            # reproject to WGS84 if necessary
            if vector_data.crs != 'EPSG:4326':
                vector_data = vector_data.to_crs(epsg=4326)
            geo_json_data = vector_data.__geo_interface__
        else:
            # if the input is a shapely geometry, wrap it in a GeoSeries and reproject if needed
            gdf = gpd.GeoSeries([vector_data], crs='EPSG:4326' if vector_data.crs is None else vector_data.crs)
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs(epsg=4326)
            geo_json_data = gdf.__geo_interface__

        for feature in geo_json_data['features']:
            feature['id'] = feature.get('id', str(feature['properties']))

        def on_click_handler(event, feature, **kwargs):
            properties = feature.get('properties', {})
            if properties:
                attr_html = "<br>".join([f"<b>{k}:</b> {v}" for k, v in properties.items()])
            else:
                attr_html = "No properties available"
            print(f"Clicked on feature with properties: {properties}")
            
            popup = Popup(
                location=event['coordinates'],
                child=HTML(attr_html),
                close_button=True,
                auto_close=True,
                close_on_escape_key=True
            )
            self.map.add_layer(popup)

        geo_json_layer = GeoJSON(
            data=geo_json_data,
            style={
                'color': color,
                'fillColor': fill_color,
                'opacity': opacity,
                'fillOpacity': fill_opacity
            },
            name=layer_name,
            hover_style={'color': 'green', 'weight': 2},
            point_to_layer=lambda i, **kwargs: {
                'radius': 8,
                'color': color,
                'fillColor': fill_color,
                'opacity': opacity,
                'fillOpacity': fill_opacity
            },
            on_click=on_click_handler  # attach the click handler
        )
        
        self.map.add_layer(geo_json_layer)

        
    def handle_draw(self, event, action, geo_json):
        if action == 'created':
            coordinates = geo_json['geometry']['coordinates'][0]
            corner1 = [str(x) for x in coordinates[0]]
            corner2 = [str(x) for x in coordinates[2]]
            
            with self.out:
                self.out.clear_output()
                print('Coordinates: ')
                print(', '.join(corner1 + corner2))
    
    def show(self):
        """Displays the map."""
        display(self.map)

    def add(self, ee_image, vis_params=None, name='New Layer', opacity=1.0, cmap=None, roi=None):
        """Adds a new Earth Engine image layer to the existing map."""
        self.add_ee_layer(ee_image, vis_params, name, opacity, cmap, roi)

# EOF
