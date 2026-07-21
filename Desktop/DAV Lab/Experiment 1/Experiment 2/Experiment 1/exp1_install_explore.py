"""
CS4503: DATA ANALYTICS AND VISUALIZATION
Experiment 1: Download, install and explore the features of
NumPy, SciPy, Jupyter, Statsmodels, Pandas, Matplotlib, Seaborn, Plotly, and Bokeh.

AIM:
To download, install, and explore the features of NumPy, SciPy, Jupyter, Statsmodels,
Pandas, Matplotlib, Seaborn, Plotly, and Bokeh for scientific computing, data analysis,
and visualization.

Install (command prompt / terminal):
    pip install numpy scipy jupyter statsmodels pandas matplotlib seaborn plotly bokeh
"""

print("=" * 60)
print("EXPERIMENT 1: LIBRARY INSTALLATION & VERSION CHECK")
print("=" * 60)

# ---- NumPy ----
import numpy as np
print("NumPy Version:", np.__version__)

# ---- Pandas ----
import pandas as pd
print("Pandas Version:", pd.__version__)

# ---- Matplotlib ----
import matplotlib
print("Matplotlib Version:", matplotlib.__version__)

# ---- Seaborn ----
import seaborn as sns
print("Seaborn Version:", sns.__version__)

# ---- SciPy ----
import scipy
print("SciPy Version:", scipy.__version__)

# ---- Statsmodels ----
try:
    import statsmodels.api as sm
    print("Statsmodels Version:", sm.__version__)
except ImportError as e:
    print("Statsmodels Version: NOT AVAILABLE in this offline sandbox ->", e)
    print("  (Install locally with: pip install statsmodels)")

# ---- Plotly ----
try:
    import plotly
    print("Plotly Version:", plotly.__version__)
except ImportError as e:
    print("Plotly Version: NOT AVAILABLE in this offline sandbox ->", e)
    print("  (Install locally with: pip install plotly)")

# ---- Bokeh ----
try:
    import bokeh
    print("Bokeh Version:", bokeh.__version__)
except ImportError as e:
    print("Bokeh Version: NOT AVAILABLE in this offline sandbox ->", e)
    print("  (Install locally with: pip install bokeh)")

# ---- JupyterLab ----
try:
    import jupyterlab
    print("JupyterLab Version:", jupyterlab.__version__)
except ImportError as e:
    print("JupyterLab Version: NOT AVAILABLE in this offline sandbox ->", e)
    print("  (Install locally with: pip install jupyterlab)")

print("\nAll available libraries successfully verified.")
print("Libraries are ready for scientific computing, data analysis, and visualization.")
