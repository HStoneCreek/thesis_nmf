from genolytics.executions.load_data import load_leukemia_data, load_barley
from synthetic_data.synthetic_uniform import generate_synthetic_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO
from matplotlib.patches import Patch
import pycountry
import pandas as pd
import geopandas as gpd

def get_data(name:str):
    if name == "leukemia":
        X, y = load_leukemia_data()
        X = np.round(np.sqrt(X))
    elif name == "barley":
        X,y = load_barley()
    elif name == "synthetic":
        X,y,_ = generate_synthetic_data()
    else:
        raise ValueError(f"Requested dataset not present. You provided: {name}")
    return X, y

def draw_pie(dist, xpos, ypos, size, ax, colors):
    # Create a pie chart as a new figure
    fig, pie_ax = plt.subplots(figsize=(.5, .5))  # Adjust figure size as needed
    pie_ax.pie(dist, colors=colors)
    pie_ax.set_aspect('equal')  # Ensure the pie chart is drawn as a circle
    # Save the pie chart to a BytesIO object
    pie_buffer = BytesIO()
    plt.savefig(pie_buffer, format='png', transparent=True, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    # Load this image into an OffsetImage
    pie_buffer.seek(0)  # Move the start to the beginning of the BytesIO object
    image = plt.imread(pie_buffer)
    imagebox = OffsetImage(image)  # Adjust zoom level as necessary
    ab = AnnotationBbox(imagebox, (xpos, ypos), frameon=False, pad=0, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)

def plot_continent(name, model, world, relative_counts):
    if name == "europe":
        world_europe = world[(world['continent'] == 'Europe') & (world['name'] != 'Russia')]
        title = "Europe"
    elif name == "near_east":
        near_east_countries = ['Turkey', 'Syria', 'Lebanon', 'Israel', 'Jordan', 'Iraq', 'Saudi Arabia', 'Yemen',
                               'Oman',
                               'United Arab Emirates', 'Qatar', 'Bahrain', 'Kuwait', 'Iran', 'Armenia', 'Georgia']
        world_europe = world[world['name'].isin(near_east_countries)]
        title = "Near East"
    elif name == 'asia':
        near_east_countries = ['Turkey', 'Syria', 'Lebanon', 'Israel', 'Jordan', 'Iraq', 'Saudi Arabia', 'Yemen',
                               'Oman',
                               'United Arab Emirates', 'Qatar', 'Bahrain', 'Kuwait', 'Iran', 'Armenia', 'Georgia']
        rest_of_asia_countries = list(set(world[world['continent'] == 'Asia']['name']) - set(near_east_countries))
        world_europe = world[world['name'].isin(rest_of_asia_countries)]
        title = "Asia"
    elif name == 'africa':
        world_europe = world[(world['continent'] == 'Africa') & (world['name'] != 'Russia')]
        title = "Africa"

    fig, ax = plt.subplots(figsize=(10, 10))
    world_europe.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=1)
    cluster_labels = relative_counts.columns
    colors = plt.cm.tab20.colors[:len(cluster_labels)]
    for idx, row in world_europe.iterrows():
        iso_code = row['iso_a3']
        if iso_code in relative_counts.index:
            proportions = relative_counts.loc[iso_code]
            centroid = row.geometry.centroid
            centroidx = centroid.x
            centroidy = centroid.y
            if iso_code == 'FRA':
                centroidx = centroid.x + 5
                centroidy = centroid.y + 5
            draw_pie(proportions, centroidx, centroidy, size=0.1, ax=ax, colors=colors)  # Adjust size as needed

    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=f'Cluster {cluster_labels[i]}') for i in
                       range(len(cluster_labels))]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=12)
    if name == "europe":
        ax.set_xlim(-12, 45)  # longitude limits (degrees East)
        ax.set_ylim(25, 75)
    if name == 'asia':
        ax.set_xlim(43, 150)

    plt.title(title, fontsize=24)
    plt.savefig(f"figs/world_map_{name}_{model}.png", dpi=300, bbox_inches='tight')
    plt.show()

country_corrections = {
            "USA": "United States",
            "Russia": "Russian Federation",
            "Bolivia": "Bolivia, Plurinational State of",
            "Iran": "Iran, Islamic Republic of",
            "Venezuela": "Venezuela, Bolivarian Republic of",
            "South Korea": "Korea, Republic of",
            "North Korea": "Korea, Democratic People's Republic of",
            "Germany/Czech Republic": "Germany",  # Take the first mentioned
            "Germany/Netherlands": "Germany",  # Take the first mentioned
            "Ex. Yugoslavia": "Yugoslavia",  # Assuming you want former Yugoslavia
            "Republic of Korea": "Korea, Republic of",
            "Finland/Sweden": "Finland",  # Take the first mentioned
            "Yugoslavia": "Serbia",  # Assuming modern equivalent, Serbia was part of former Yugoslavia
            "Tibet": "China"
        }

def get_iso_code(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None

def get_table(category, X):
    row_type = pd.pivot_table(X[["clusters", category]], index=category, columns='clusters', aggfunc=len,
                              fill_value=0)

    # Adding row totals
    row_type['Total'] = row_type.sum(axis=1)

    # Adding column totals
    row_type.loc['Total', :] = row_type.sum()
    print(row_type)

def analyze_barley_data(predicted_clusters):
    X, y = load_barley(drop_meta=False)

    value_counts = pd.Series(predicted_clusters).value_counts()

    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    mapping = {value: letter for value, letter in zip(value_counts.index[:6], letters)}

    s_replaced = pd.Series(predicted_clusters).replace(mapping)

    X['clusters'] = s_replaced
    X = X.sort_values("clusters")

    X["Row_type"] = X["Row_type"].replace(np.nan, "Other")
    X["Breeding History"] = X["Breeding History"].replace(np.nan, "Other")
    X["Growth habit"] = X["Growth habit"].replace(np.nan, "Other")
    # Creating a new column for easier pivoting (combining all classifications)
    X['Classification'] = X['Row_type'].astype(str) + " | " + X['Breeding History'] + " | " + X['Growth habit']

    # Pivot table to count occurrences of cluster assignments within each classification
    pivot_table = pd.pivot_table(X[["clusters", "Classification"]], index='Classification', columns='clusters',
                                 aggfunc=len,
                                 fill_value=0)

    get_table("Row_type", X)
    get_table("Breeding History", X)
    get_table("Growth habit", X)

    X = X.sort_values("clusters")
    print(X["clusters"].value_counts())

    X['Country'] = X['Country'].apply(lambda x: country_corrections.get(x, x))
    X['iso_code'] = X["Country"].apply(get_iso_code)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world.to_crs(epsg=4326)
    # Group by ISO code and cluster to count occurrences
    cluster_counts = X.groupby(['iso_code', 'clusters']).size().reset_index(name='counts')
    # Suppose cluster_counts is already defined
    relative_counts = cluster_counts.pivot(index='iso_code', columns='clusters', values='counts').fillna(0)
    relative_counts = relative_counts.div(relative_counts.sum(axis=1), axis=0)  # Normalize to get proportions

    # Determine the majority cluster by country
    majority_clusters = cluster_counts.loc[cluster_counts.groupby('iso_code')['counts'].idxmax()]

    # Merge the majority cluster back onto the world GeoDataFrame
    world = world.merge(majority_clusters, how='left', left_on='iso_a3', right_on='iso_code')

    # Create a categorical type for clusters for consistent coloring
    world['clusters'] = pd.Categorical(world['clusters'])

    plot_continent('europe', model="NMF", world=world, relative_counts=relative_counts)
    plot_continent('africa', model="NMF", world=world, relative_counts=relative_counts)
    plot_continent('near_east', model="NMF", world=world, relative_counts=relative_counts)
    plot_continent('asia', model="NMF", world=world, relative_counts=relative_counts)

