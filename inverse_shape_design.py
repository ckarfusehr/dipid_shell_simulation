import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############################################
# 1) Feature Extraction (same as your code)
############################################

def extract_features(point_cloud):
    """
    Extract global shape descriptors from a 3D point cloud:
    PCA-based eigenvalue ratios, aspect ratios, convex hull info,
    skewness, kurtosis, point density, etc.
    """
    centroid = point_cloud.mean(axis=0)
    centered_points = point_cloud - centroid

    # PCA-based features
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending order

    # Avoid division by zero
    if eigenvalues.sum() == 0:
        pca_ratios = [0, 0, 0]
    else:
        pca_ratios = eigenvalues / eigenvalues.sum()

    aspect_ratios = [
        eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 0,
        eigenvalues[1] / eigenvalues[2] if eigenvalues[2] > 0 else 0,
        eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 0 else 0,
    ]

    # Convex Hull
    try:
        hull = ConvexHull(centered_points)
        hull_volume = hull.volume
        hull_surface_area = hull.area
        compactness = hull_volume / (hull_surface_area ** 3) if hull_surface_area > 0 else 0
    except:
        hull_volume = 0
        hull_surface_area = 0
        compactness = 0

    # Distribution features
    try:
        skewness_vals = skew(centered_points, axis=0, nan_policy='omit')
        kurt_values = kurtosis(centered_points, axis=0, nan_policy='omit')
    except:
        skewness_vals = [0, 0, 0]
        kurt_values = [0, 0, 0]

    # Point density
    point_density = len(point_cloud) / hull_volume if hull_volume > 0 else 0

    # Compile features into a dictionary
    features = {
        "eigenvalue_ratio_1": pca_ratios[0],
        "eigenvalue_ratio_2": pca_ratios[1],
        "eigenvalue_ratio_3": pca_ratios[2],
        "aspect_ratio_major_minor1": aspect_ratios[0],
        "aspect_ratio_minor1_minor2": aspect_ratios[1],
        "aspect_ratio_major_minor2": aspect_ratios[2],
        "hull_volume": hull_volume,
        "hull_surface_area": hull_surface_area,
        "compactness": compactness,
        "skewness_x": skewness_vals[0],
        "skewness_y": skewness_vals[1],
        "skewness_z": skewness_vals[2],
        "kurtosis_x": kurt_values[0],
        "kurtosis_y": kurt_values[1],
        "kurtosis_z": kurt_values[2],
        "point_density": point_density,
    }
    return features


###############################################
# 2) Aggregate features by alpha and get targets
###############################################

def get_mean_features_for_alpha(feature_df, a_values, alpha):
    """
    Given:
      - feature_df: DataFrame of shape features (one row per simulation).
      - a_values: list of alpha values, one per row in feature_df (same length).
      - alpha: the alpha value for which we want the average features.
    
    Returns a dict of mean features across all simulations with that alpha.
    """
    # Find rows corresponding to alpha
    indices = [i for i, a_val in enumerate(a_values) if a_val == alpha]
    subset = feature_df.iloc[indices]

    if subset.empty:
        raise ValueError(f"No features found for alpha={alpha}.")

    # Compute mean (you could also try median or other robust statistics)
    mean_series = subset.mean(numeric_only=True)
    target_features = mean_series.to_dict()  # convert to a dict
    return target_features


###############################################
# 3) Parametric Shape Model (sphere + offsets)
###############################################

def generate_fibonacci_sphere(n_points=200, seed=42):
    """
    Generate a fixed set of angles (phi, theta) or 3D directions on a sphere
    via the Fibonacci spiral approach, returning Nx2 array of (phi, theta).

    We'll store spherical coords:
      phi   in [0, 2pi),   (azimuth)
      theta in [0,  pi],   (polar angle from z-axis)
    """
    np.random.seed(seed)
    golden_ratio = (1 + 5 ** 0.5) / 2
    angles = np.zeros((n_points, 2))

    for i in range(n_points):
        frac = (i + 0.5) / n_points
        theta = np.arccos(1 - 2 * frac)  # polar angle
        phi = 2 * np.pi * i / golden_ratio
        # store them
        angles[i, 0] = phi
        angles[i, 1] = theta
    return angles

def shape_to_point_cloud(params, angles):
    """
    Given:
      - params: radial offsets for each point (length = len(angles))
      - angles: Nx2 array with [phi, theta] for each point

    Returns Nx3 array of points in 3D.
    Spherical coords:
      x = r * sin(theta) * cos(phi)
      y = r * sin(theta) * sin(phi)
      z = r * cos(theta)
    """
    assert len(params) == len(angles), "params and angles must have the same length."
    point_cloud = np.zeros((len(params), 3))

    for i in range(len(params)):
        phi = angles[i, 0]
        theta = angles[i, 1]
        r = params[i]

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        point_cloud[i] = [x, y, z]

    return point_cloud


###############################################
# 4) Objective Function for Optimization
###############################################

def objective_function(params, angles, target_features):
    """
    1) Generate a point cloud from params + angles
    2) Extract features
    3) Compare to target_features
    4) Return MSE (sum of squared errors / # features)
    """
    pc = shape_to_point_cloud(params, angles)
    feats = extract_features(pc)

    mse = 0.0
    n_keys = 0
    for k in target_features:
        # avoid potential mismatch with missing keys
        if k in feats:
            diff = feats[k] - target_features[k]
            mse += diff * diff
            n_keys += 1

    if n_keys > 0:
        mse /= n_keys
    else:
        mse = 9999.0  # penalty if no matching features

    return mse


###############################################
# 5) Fit One Representative Shape for a Given α
###############################################

def fit_representative_shape_for_alpha(feature_df, a_values, alpha,
                                       n_points=200, 
                                       r_min=0.5, r_max=1.5,
                                       max_iter=50, popsize=15, seed=42):
    """
    1) Compute mean feature vector for all structures at given alpha.
    2) Optimize radial offsets for parametric sphere to match these mean features.
    3) Return the resulting point cloud + final metrics.
    """

    # --- (A) Get target features for alpha
    target_features = get_mean_features_for_alpha(feature_df, a_values, alpha)

    # --- (B) Prepare a fixed set of angles (sphere sampling)
    angles = generate_fibonacci_sphere(n_points=n_points, seed=seed)

    # We'll have one radial parameter per point, so that's n_points parameters.
    bounds = [(r_min, r_max)] * n_points

    # --- (C) Optimize using differential evolution
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(angles, target_features),
        maxiter=max_iter,
        popsize=popsize,
        seed=seed,
        polish=True,
        disp=False
    )

    # --- (D) Generate final shape
    best_params = result.x
    reconstructed_cloud = shape_to_point_cloud(best_params, angles)

    # --- (E) Compute final metrics
    final_error = objective_function(best_params, angles, target_features)
    final_features = extract_features(reconstructed_cloud)

    # Put together a quality dictionary
    quality_dict = {
        'alpha': alpha,
        'final_objective': final_error,
    }
    # Also store absolute differences for each feature
    for k in target_features:
        if k in final_features:
            quality_dict[f'diff_{k}'] = final_features[k] - target_features[k]

    return reconstructed_cloud, quality_dict

###############################################
# 6) Visualization in Jupyter
###############################################

def visualize_point_cloud_3d(point_cloud, title="Reconstructed Shape"):
    """
    Creates a 3D scatter plot of the given point cloud.
    This should allow interactive rotation if using:
        %matplotlib notebook
    or 
        %matplotlib widget
    in a Jupyter environment.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
               s=15, c='blue', alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()
    
import plotly.graph_objects as go

def visualize_point_cloud_plotly(point_cloud, title="Reconstructed Shape"):
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    
    fig = go.Figure(
        data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3))]
    )
    fig.update_layout(
        title=title, 
        scene=dict(
            xaxis_title='X', 
            yaxis_title='Y', 
            zaxis_title='Z'
        )
    )
    fig.show()

# Usage:
#visualize_point_cloud_plotly(rep_cloud, title=f"Shape for alpha={alpha_val}")


###############################################
# 7) Example Usage and Demo
###############################################
if __name__ == "__main__":
    # For demonstration, let's create some synthetic feature data
    #   Just 3 random "structures" for alpha=1.0 and 3 for alpha=2.0
    np.random.seed(0)
    data = []
    a_list = []

    # Fake data for alpha=1.0
    for i in range(3):
        data.append({
            "eigenvalue_ratio_1": np.random.rand(),
            "eigenvalue_ratio_2": np.random.rand(),
            "eigenvalue_ratio_3": np.random.rand(),
            "aspect_ratio_major_minor1": np.random.rand(),
            "aspect_ratio_minor1_minor2": np.random.rand(),
            "aspect_ratio_major_minor2": np.random.rand(),
            "hull_volume": np.random.rand() * 50,
            "hull_surface_area": np.random.rand() * 100,
            "compactness": np.random.rand(),
            "skewness_x": np.random.randn(),
            "skewness_y": np.random.randn(),
            "skewness_z": np.random.randn(),
            "kurtosis_x": np.random.randn(),
            "kurtosis_y": np.random.randn(),
            "kurtosis_z": np.random.randn(),
            "point_density": np.random.rand() * 10,
        })
        a_list.append(1.0)

    # Fake data for alpha=2.0
    for i in range(3):
        data.append({
            "eigenvalue_ratio_1": np.random.rand(),
            "eigenvalue_ratio_2": np.random.rand(),
            "eigenvalue_ratio_3": np.random.rand(),
            "aspect_ratio_major_minor1": np.random.rand(),
            "aspect_ratio_minor1_minor2": np.random.rand(),
            "aspect_ratio_major_minor2": np.random.rand(),
            "hull_volume": np.random.rand() * 50,
            "hull_surface_area": np.random.rand() * 100,
            "compactness": np.random.rand(),
            "skewness_x": np.random.randn(),
            "skewness_y": np.random.randn(),
            "skewness_z": np.random.randn(),
            "kurtosis_x": np.random.randn(),
            "kurtosis_y": np.random.randn(),
            "kurtosis_z": np.random.randn(),
            "point_density": np.random.rand() * 10,
        })
        a_list.append(2.0)

    feature_df = pd.DataFrame(data)
    a_values = np.array(a_list)

    # --- (A) Fit a representative shape for alpha=1.0
    alpha_val = 1.0
    rep_cloud, quality = fit_representative_shape_for_alpha(
        feature_df=feature_df, 
        a_values=a_values, 
        alpha=alpha_val,
        n_points=100,   # number of sample points on the shape
        r_min=0.5,      # minimum radial offset
        r_max=1.5,      # maximum radial offset
        max_iter=30,    # iteration count for differential evolution
        popsize=10,
        seed=42
    )

    print(f"Fitted shape for alpha={alpha_val} has final objective: {quality['final_objective']:.6f}")
    print("Feature differences:", 
          {k: v for k, v in quality.items() if k.startswith('diff_')})

    # The 'rep_cloud' is the Nx3 array of your representative shape.
    # You can also re-run extract_features(rep_cloud) to see final shape descriptors.
    final_feats = extract_features(rep_cloud)
    print("Final shape's features:", final_feats)

 
