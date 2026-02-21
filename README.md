# PLM2D: Partially Linked Multi-Matrix Decomposition
To greatly reduce the high dimension and adequately describe the complicated correlation of multivariate heterogeneous profile data, a joint matrix decomposition is applied to the high-dimensional observations of
all profiles to uncover their low-rank latent factors, and more importantly, the scores of these factors
are grouped into three kinds of components: 

- globally common ones that are shared by all profiles,
- partially common ones shared by some but not all profiles, and
- individual components specific only to one profile.

It is described in
- "Common Components or Individual Ones or Between? A Partially Linked Multi-Matrix Decomposition for Joint Modeling of Multivariate Heterogeneous Profile Data", Runyu Mao, Kai Wang*, Chen Zhang, Fugee Tsung.

## Repository Structure
- `plmmd.py`: Core implementation of the PLM2D algorithm.
- `benchmarks.py`: Script to execute all benchmark comparisons in simulations.
- `monitoring.py`: Script for process monitoring.
- `data_generation.py`: Data simulation engine for generating both in-control and out-of-control scenarios.
- `simulation_plots.py` & `mtr_visualization.py`: Visualization tools for reproducing figures in the paper.
- `eigenvectors/`: Directory containing MTR loading matrices from for generating simulated data.
- `structure_5stations.csv` & `structure_5stations_original.csv`: The pre-defined cluster structure used in simulations.
- `illustrative_example.py`: An illustrative example considering multiple passenger flow profiles collected at all stations of the Island Line of Hong Kong
Mass Transit Railway (MTR) system..

## Dependencies
The code is written in Python 3.11. 
