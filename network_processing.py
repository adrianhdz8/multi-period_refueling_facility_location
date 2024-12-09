"""
DOCSTRING TEXT
"""
from util import *


# LOAD STORED GRAPHS


def load_simplified_consolidated_graph(rr: str) -> nx.Graph:

    gpkl_path = os.path.join(NX_DIR, rr + '_geo_graph_simplified.pkl')
    if os.path.exists(gpkl_path):
        G = nx.read_gpickle(gpkl_path)
    else:
        # TODO: return <ErrorMessage>?
        G = None
        # nx.write_gpickle(G, gpkl_path)

    return G


# GRAPH COORDINATE PROJECTION


def project_graph(G, to_crs=None, smooth_geometry=False, smooth_geometry_tolerance=0.05):
    """
    Project graph from its current CRS to another.

    If to_crs is None, project the graph to the UTM CRS for the UTM zone in
    which the graph's centroid lies. Otherwise, project the graph to the CRS
    defined by to_crs.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        the graph to be projected
    to_crs : string or pyproj.CRS
        if None, project graph to UTM zone in which graph centroid lies,
        otherwise project graph to this CRS

    Returns
    -------
    G_proj : networkx.MultiDiGraph
        the projected graph
    """

    # if isinstance(G.graph['crs'], str):
    #     if G.graph['crs'] == to_crs:
    #         return G
    # elif G.graph['crs'].name.replace(' ', '') == to_crs:
    #     return G
    #
    # # STEP 1: PROJECT THE NODES
    # gdf_nodes = gdfs_from_graph(G, edges=False)
    #
    # # create new lat/lng columns to preserve lat/lng for later reference if
    # # cols do not already exist (ie, don't overwrite in later re-projections)
    # if "lon" not in gdf_nodes.columns or "lat" not in gdf_nodes.columns:
    #     gdf_nodes["lon"] = gdf_nodes["x"]
    #     gdf_nodes["lat"] = gdf_nodes["y"]
    #
    # # project the nodes GeoDataFrame and extract the projected x/y values
    # gdf_nodes_proj = ox.project_gdf(gdf_nodes, to_crs=to_crs)
    # gdf_nodes_proj["x"] = gdf_nodes_proj["geometry"].x
    # gdf_nodes_proj["y"] = gdf_nodes_proj["geometry"].y
    # # gdf_nodes_proj = gdf_nodes_proj.drop(columns=["geometry"])
    #
    # # STEP 2: PROJECT THE EDGES
    # # if "simplified" in G.graph and G.graph["simplified"]:
    # #     # if graph has previously been simplified, project the edge geometries
    # gdf_edges = gdfs_from_graph(G, nodes=False)
    # gdf_edges_proj = ox.project_gdf(gdf_edges, to_crs=to_crs)

    gdf_nodes_proj, gdf_edges_proj = gdfs_from_graph(G, crs=to_crs, smooth_geometry=smooth_geometry,
                                                     smooth_geometry_tolerance=smooth_geometry_tolerance)
    # STEP 3: REBUILD GRAPH
    # turn projected node/edge gdfs into a graph and update its CRS attribute
    if G.is_directed():
        G_proj = digraph_from_gdfs(gdf_nodes_proj, gdf_edges_proj, G.graph)
    else:
        G_proj = undirected_graph_from_gdfs(gdf_nodes_proj, gdf_edges_proj, G.graph)
    G_proj.graph["crs"] = gdf_nodes_proj.crs

    return G_proj


def project_nodes_gdf(nodes_gdf: gpd.GeoDataFrame, to_crs: str = 'WGS84'):

    if nodes_gdf.crs.name.replace(' ', '') == to_crs:
        return nodes_gdf

    # PROJECT THE NODES
    # create new lat/lng columns to preserve lat/lng for later reference if
    # cols do not already exist (ie, don't overwrite in later re-projections)
    # gdf_nodes_proj = gdf_nodes_proj.drop(columns=["geometry"])

    if "lon" not in nodes_gdf.columns or "lat" not in nodes_gdf.columns:
        nodes_gdf["lon"] = nodes_gdf["x"]
        nodes_gdf["lat"] = nodes_gdf["y"]

    # project the nodes GeoDataFrame and extract the projected x/y values
    nodes_gdf_proj = ox.project_gdf(nodes_gdf, to_crs=to_crs)
    nodes_gdf_proj["x"] = nodes_gdf_proj["geometry"].x
    nodes_gdf_proj["y"] = nodes_gdf_proj["geometry"].y

    return nodes_gdf_proj


def project_edges_gdf(edges_gdf: gpd.GeoDataFrame, to_crs: str = 'WGS84'):

    # PROJECT THE EDGES
    # if "simplified" in G.graph and G.graph["simplified"]:
    #     # if graph has previously been simplified, project the edge geometries
    # edges_gdf = gdfs_from_graph(G, nodes=False)

    if edges_gdf.crs.name.replace(' ', '') == to_crs:
        return edges_gdf

    edges_gdf_proj = ox.project_gdf(edges_gdf, to_crs=to_crs)

    return edges_gdf_proj


def remove_from_graph(G: nx.Graph, nodes_to_remove=None, edges_to_remove=None, connected_only=False) -> nx.Graph:
    G = G.copy()

    if nodes_to_remove is None:
        nodes_to_remove = []
    if edges_to_remove is None:
        edges_to_remove = []

    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(nodes_to_remove)

    if connected_only:
        # extract largest connected component from graph
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    return G


# CONVERSION: GEODATAFRAMES <-> GRAPHS


def gdfs_from_graph(G: nx.Graph, nodes=True, edges=True, crs: str = 'WGS84', smooth_geometry=False,
                    smooth_geometry_tolerance=0.05):
    # smooth_geometry_tolerance default unit is in kilometers?

    if nodes and not edges:
        return node_gdf_from_graph(G, crs=crs)
    elif not nodes and edges:
        return edge_gdf_from_graph(G, crs=crs, smooth_geometry=smooth_geometry,
                                   smooth_geometry_tolerance=smooth_geometry_tolerance)

    return node_gdf_from_graph(G), edge_gdf_from_graph(G, crs=crs, smooth_geometry=smooth_geometry,
                                                       smooth_geometry_tolerance=smooth_geometry_tolerance)


def node_gdf_from_graph(G: nx.Graph, crs: str = 'WGS84'):
    G = G.copy()

    nodes, data = zip(*G.nodes(data=True))
    if all(['x' in d.keys() for d in data]) and all(['y' in d.keys() for d in data]):
        node_geometry = [Point(d['x'], d['y']) for d in data]
        gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes, geometry=node_geometry)
        if "crs" in G.graph.keys():
            gdf_nodes.crs = G.graph["crs"]
    else:
        gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes)

    return project_nodes_gdf(nodes_gdf=gdf_nodes, to_crs=crs)


def edge_gdf_from_graph(G: nx.Graph, crs: str = 'WGS84', smooth_geometry=True, smooth_geometry_tolerance=0.05):
    # smooth_geometry_tolerance default unit is in kilometers?

    G = G.copy()

    for e in G.edges():
        u = e[0]
        v = e[1]
        G.edges[u, v]['u'] = u
        G.edges[u, v]['v'] = v

    if G.number_of_edges() != 0:
        _, _, edge_data = zip(*G.edges(data=True))
        gdf_edges = gpd.GeoDataFrame(list(edge_data))
        gdf_edges = gdf_edges.groupby(by=['u', 'v'], as_index=True).first()
        if 'geometry' in edge_data[0].keys() and 'crs' in G.graph.keys():
            gdf_edges.crs = G.graph['crs']
    else:
        gdf_edges = gpd.GeoDataFrame()

    if smooth_geometry and 'geometry' in gdf_edges.columns:
        gdf_edges['geometry'] = gdf_edges['geometry'].simplify(smooth_geometry_tolerance)

    return project_edges_gdf(edges_gdf=gdf_edges, to_crs=crs)


def undirected_graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None) -> nx.Graph:
    """
    Convert nodes and links geodataframes into a network representation using netwokrx
    :param rr: [str] railroad of interest; will remove all other railroads
    :return: [nx.MultiGraph] undirected graph with node id's used to reference nodes and edges;
                        features from geodataframes will be stored in the graph node and edge attributes
    """

    if graph_attrs is None:
        graph_attrs = {"crs": gdf_edges.crs}
    Gun = nx.Graph(**graph_attrs)

    node_cols = gdf_nodes.columns
    edge_cols = gdf_edges.columns
    # for each link entry in the link geodataframe; add new edges and nodes with information
    for u, v in gdf_edges.index:
        edge_data = gdf_edges.loc[(u, v)]  # gdf row of edge e's features
        u_data = gdf_nodes.loc[u]  # gdf row of node u's features
        v_data = gdf_nodes.loc[v]  # gdf row of node v's features

        # check if nodes have not already been added
        if u not in Gun:
            Gun.add_nodes_from([(u, {a: u_data[a] for a in node_cols})])
        if v not in Gun:
            Gun.add_nodes_from([(v, {a: v_data[a] for a in node_cols})])

        # features to add as edge attributes
        Gun.add_edges_from([(u, v, {a: edge_data[a] for a in edge_cols})])

    return Gun


def digraph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None) -> nx.DiGraph:
    """
    Convert node and edge GeoDataFrames to a MultiDiGraph.

    This function is the inverse of `graph_to_gdfs` and is designed to work in
    conjunction with it. However, you can convert arbitrary node and edge
    GeoDataFrames as long as gdf_nodes is uniquely indexed by `osmid` and
    gdf_edges is uniquely multi-indexed by `u`, `v`, `key` (following normal
    MultiDiGraph structure). This allows you to load any node/edge shapefiles
    or GeoPackage layers as GeoDataFrames then convert them to a MultiDiGraph
    for graph analysis.

    Parameters
    ----------
    gdf_nodes : geopandas.GeoDataFrame
        GeoDataFrame of graph nodes uniquely indexed by osmid
    gdf_edges : geopandas.GeoDataFrame
        GeoDataFrame of graph edges uniquely multi-indexed by u, v, key
    graph_attrs : dict
        the new G.graph attribute dict. if None, use crs from gdf_edges as the
        only graph-level attribute (gdf_edges must have crs attribute set)

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    if gdf_nodes is not None and gdf_edges is not None:
        if graph_attrs is None:
            graph_attrs = {"crs": gdf_edges.crs}
        G = nx.DiGraph(**graph_attrs)

        # add edges and their attributes to graph, but filter out null attribute
        # values so that edges only get attributes with non-null values
        attr_names = gdf_edges.columns.to_list()
        for (u, v), attr_vals in zip(gdf_edges.index, gdf_edges.values):
            data_all = zip(attr_names, attr_vals)
            data = {name: val for name, val in data_all if isinstance(val, list) or pd.notnull(val)}
            G.add_edge(u, v, **data)

        # add nodes' attributes to graph
        for col in gdf_nodes.columns:
            nx.set_node_attributes(G, name=col, values=gdf_nodes[col].dropna())

        return G

    return None


# MISC. METHODS


def augment_stations_splc(G: nx.Graph):
    """
    Augment G's node attributes for SPLC's with additional SPLC data from <filename> file
    by assigning to geographically closest nodes on G
    """

    G = G.copy()

    df_splc = load_splc_station_master()

    gdf_splc = gpd.GeoDataFrame(df_splc, geometry=gpd.points_from_xy(df_splc['lon'], df_splc['lat'],
                                                                     crs=G.graph['crs']))

    gdf_nodes, gdf_edges = gdfs_from_graph(G)

    # project to a 2-D projection prior this
    crs = 'epsg:5070'
    gdf_splc.to_crs(crs=crs, inplace=True)
    gdf_nodes.to_crs(crs=crs, inplace=True)
    gdf_join = gpd.sjoin_nearest(gdf_splc.to_crs(crs=crs), gdf_nodes.to_crs(crs=crs),
                                 how='left', max_distance=5e5, distance_col='distance')
    gdf_join.to_crs(crs=G.graph['crs'], inplace=True)
    gdf_join.dropna(inplace=True)

    gdf_join['SPLC'] = gdf_join['SPLC'].astype(int)
    gdf_join['SPLC'] = gdf_join['SPLC'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))
    gdf_join.index = gdf_join['SPLC']
    gdf_join.drop_duplicates(subset='SPLC', inplace=True)

    for splc in gdf_join.index:
        node = gdf_join.loc[splc, 'nodeid']
        splc_set = set(G.nodes[node]['splc'])
        G.nodes[node]['splc'] = list(splc_set.union(set([splc])))

    return G


def augment_node_city_name(G: nx.Graph):
    """
    Augment G's node attributes for SPLC's with additional SPLC data from <filename> file
    by assigning to geographically closest nodes on G
    """

    G = G.copy()

    df_city = load_us_cities()

    gdf_city = gpd.GeoDataFrame(df_city, geometry=gpd.points_from_xy(df_city['lng'], df_city['lat'],
                                                                     crs=G.graph['crs']))

    gdf_nodes, gdf_edges = gdfs_from_graph(G)

    # project to a 2-D projection prior this
    crs = 'epsg:5070'
    # gdf_city.to_crs(crs=crs, inplace=True)
    # gdf_nodes.to_crs(crs=crs, inplace=True)
    gdf_join = gpd.sjoin_nearest(gdf_nodes.to_crs(crs=crs), gdf_city.to_crs(crs=crs),
                                 how='left', max_distance=5e5, distance_col='distance')
    gdf_join.to_crs(crs=G.graph['crs'], inplace=True)
    # gdf_join.dropna(inplace=True)

    # gdf_join.index = gdf_join['SPLC']
    # gdf_join.drop_duplicates(subset='SPLC', inplace=True)

    for node in gdf_join.index:
        G.nodes[node]['city'] = gdf_join.loc[node, 'city']
        G.nodes[node]['state'] = gdf_join.loc[node, 'state_id']

    return G


def smooth_base_graph_geometry(rr: str, smooth_geometry_tolerance=0.05):

    gpkl_original_path = os.path.join(NX_DIR, rr + '_geo_graph_simplified_original.pkl')
    if os.path.exists(gpkl_original_path):
        G = nx.read_gpickle(gpkl_original_path)

        for u, v in G.edges:
            G.edges[u, v]['geometry'] = G.edges[u, v]['geometry'].simplify(smooth_geometry_tolerance)

        nx.write_gpickle(G, os.path.join(NX_DIR, rr + '_geo_graph_simplified.pkl'))

    return G
