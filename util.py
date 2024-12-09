"""
Utils
=====

Imports all required packages and sets global variables for file directories.
"""
# GENERAL
import os
import os.path
import math
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from copy import deepcopy
from itertools import product, chain, combinations, permutations, islice
from datetime import date, datetime
import time
import pickle as pkl
import json
import geojson
from tqdm import tqdm
from multiprocess import pool
# OPTIMIZATION
import gurobipy as gp
from gurobipy import GRB
# NETWORKS
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point, Polygon, MultiLineString, shape
from shapely.ops import linemerge
from shapely.errors import ShapelyDeprecationWarning
# PLOTTING
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly
import plotly.graph_objs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import iplot
import logging

# CONFIGURATIONS
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

sns.set_theme(palette='Set2')

pio.renderers.default = "browser"
tqdm.pandas()

# GLOBAL PATHS
BASE_DIR = os.path.dirname(__file__)

# data/input directory
DATA_DIR = os.path.join(BASE_DIR, 'data')
# data/input subdirectories
NX_DIR = os.path.join(DATA_DIR, 'networks')
FAF5_DIR = os.path.join(DATA_DIR, 'flows')
MAT_DIR = os.path.join(DATA_DIR, 'matrices')
PARAM_DIR = os.path.join(DATA_DIR, 'parameters')
SCENARIO_DIR = os.path.join(DATA_DIR, 'scenario')
SOLVER_PARAM_DIR = os.path.join(DATA_DIR, 'solver parameters')
FACILITY_DIR = os.path.join(DATA_DIR, 'facility')
SHORTEST_PATH_DIR = os.path.join(NX_DIR, 'distances')

# results/output directory
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
# results/output subdirectories
EXP_DIR = os.path.join(RESULTS_DIR, 'experiments')
FIGURE_DIR = os.path.join(RESULTS_DIR, 'figures')
GRB_MODEL_DIR = os.path.join(RESULTS_DIR, 'gurobi models')

# GLOBAL VARS
KM2MI = 0.62137119  # miles / km
