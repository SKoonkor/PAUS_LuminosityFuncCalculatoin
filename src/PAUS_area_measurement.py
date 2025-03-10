import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.io import ascii, fits
from scipy.spatial import cKDTree

def integral_area(dec_min, dec_max, ra_min, ra_max):
    """
    Calculate the area of a spherical rectangle defined by RA and Dec limits.
    """
    theta_term = np.deg2rad(np.abs(ra_max - ra_min))
    phi_term = np.abs(np.sin(np.deg2rad(dec_max)) - np.sin(np.deg2rad(dec_min)))
    return theta_term * phi_term * 41253 / (4 * np.pi)

def random_points_in_survey(n_points, ra_min, ra_max, dec_min, dec_max):
    """
    Generate random points uniformly distributed over the spherical patch
    defined by ra_min, ra_max, dec_min, dec_max.
    """
    # Uniformly sample RA
    ra_random = np.random.uniform(ra_min, ra_max, size=n_points)
    # For dec, sample uniformly in sin(dec)
    sin_dec_min, sin_dec_max = np.sin(np.deg2rad(dec_min)), np.sin(np.deg2rad(dec_max))
    sin_dec_random = np.random.uniform(sin_dec_min, sin_dec_max, size=n_points)
    dec_random = np.rad2deg(np.arcsin(sin_dec_random))
    return ra_random, dec_random

def radec_to_cartesian(ra, dec):
    """
    Convert RA and Dec (in degrees) to Cartesian coordinates on the unit sphere.
    """
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.vstack((x, y, z)).T

def measure_survey_area(catalog, selection, dp=0.01, n_points=1000000,
                        ra_col='alpha_j2000', dec_col='delta_j2000'):
    """
    Measure the survey area for the catalog based on the selection criteria.
    
    Returns:
      (survey_boundary_area, measured_area)
    """
    # Apply selection
    selected_catalog = catalog[selection].copy()
    
    # Determine the survey boundary
    ra_min, ra_max = selected_catalog[ra_col].min(), selected_catalog[ra_col].max()
    dec_min, dec_max = selected_catalog[dec_col].min(), selected_catalog[dec_col].max()
    survey_boundary_area = integral_area(dec_min, dec_max, ra_min, ra_max)
    
    print(f"Survey boundary: RA [{ra_min:.2f}, {ra_max:.2f}], Dec [{dec_min:.2f}, {dec_max:.2f}]")
    print(f"Boundary area: {survey_boundary_area:.5f} deg^2")
    
    # Build KDTree using 3D Cartesian coordinates
    cat_coords = radec_to_cartesian(selected_catalog[ra_col].values, selected_catalog[dec_col].values)
    kd_tree = cKDTree(cat_coords)
    
    # Generate random points within survey boundary
    ra_rand, dec_rand = random_points_in_survey(n_points, ra_min, ra_max, dec_min, dec_max)
    rand_coords = radec_to_cartesian(ra_rand, dec_rand)
    
    # Convert dp (in degrees) to chord distance on unit sphere
    dp_rad = np.deg2rad(dp)
    chord_threshold = 2 * np.sin(dp_rad / 2)
    
    # Query the KDTree for nearest neighbors
    distances, _ = kd_tree.query(rand_coords, k=1, n_jobs=-1)
    condition = distances <= chord_threshold
    n_hits = np.count_nonzero(condition)
    
    measured_area = n_hits * survey_boundary_area / n_points
    print(f"Measured survey area: {measured_area:.5f} deg^2 (hits: {n_hits} out of {n_points} points)")
    
    return survey_boundary_area, measured_area, ra_rand, dec_rand, condition

def process_survey(input_file, selection_func, dp=0.01, n_points=1000000,
                   ra_col='alpha_j2000', dec_col='delta_j2000'):
    """
    General processing function for a survey catalog.
    
    Parameters:
      input_file    : Path to the catalog CSV file.
      selection_func: Function that accepts a DataFrame and returns a boolean mask.
      dp            : Angular tolerance in degrees.
      n_points      : Number of random points for Monte Carlo integration.
      ra_col, dec_col: Column names for RA and Dec.
      
    The function saves two output CSV files:
      - {input_basename}_area_random_points.csv : Contains random points and hit condition.
      - {input_basename}_area_parameters.csv   : Contains the survey boundary area, measured area, and number of random points.
      
    Returns:
      (survey_boundary_area, measured_area)
    """
    # Read the catalog
    catalog = pd.read_csv(input_file)
    
    # Apply the selection function provided
    selection = selection_func(catalog)
    
    # Measure the survey area
    boundary_area, measured_area, ra_rand, dec_rand, condition = measure_survey_area(
        catalog, selection, dp=dp, n_points=n_points, ra_col=ra_col, dec_col=dec_col)
    
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.dirname(input_file)
    
    # Extract selection function name
    selection_name = selection_func.__name__.replace("selection_", "")
    
    random_points_csv = os.path.join(output_dir, f"{base_name}_{selection_name}_random_points.csv")
    area_params_csv = os.path.join(output_dir, f"{base_name}_{selection_name}_parameters.csv")
    
    
    # Save random point data
    random_df = pd.DataFrame({
        'ra_random': ra_rand,
        'dec_random': dec_rand,
        'condition': condition.astype(int)
    })
    random_df.to_csv(random_points_csv, index=False)
    
    # Save area parameters
    area_df = pd.DataFrame({
        'survey_boundary_area': [boundary_area],
        'measured_area': [measured_area],
        'n_random_points': [n_points]
    })
    area_df.to_csv(area_params_csv, index=False)
    
    print(f"Output saved as:\n  {random_points_csv}\n  {area_params_csv}")
    return boundary_area, measured_area








# Selection functions for different survey cuts
def W1_1044_selection_LFcal(catalog):
    """
    Selection function for CFHT_1044 survey.
    Modify the criteria as needed.
    """
    return ((catalog.mag_i >= 0) &
        (catalog.mag_g >= 0) &
        (catalog.mag_r >= 0) &
        (catalog.mag_u >= 0) &
        (catalog.mag_z >= 0) &
        (catalog.mag_i <= 23) &
        (catalog.mag_u <= 50) &
        (catalog.mag_g <= 50) &
        (catalog.mag_r <= 50) &
        (catalog['mask'] <= 1) &
        (catalog.mag_z <= 50) &
        (catalog.star_flag == 0) &
        (catalog.qz <= 1e4)
    )

def W3_1045_selection_LFcal(catalog):
    """
    Example selection function for W3_1045 survey.
    Adjust the criteria for W3 as needed.
    """
    return ((catalog.mag_i >= 0) &
             (catalog.mag_g >= 0) &
             (catalog.mag_r >= 0) &
             (catalog.mag_u >= 0) &
             (catalog.mag_z >= 0) &
             (catalog.mag_i <= 24) &
             (catalog.mag_u <= 50) &
             (catalog.mag_g <= 50) &
             (catalog.mag_r <= 50) &
             (catalog['mask'] <= 1) &
             (catalog.mag_z <= 50) & 
             (catalog.star_flag == 0) & 
             (catalog.lp_mi >= -30)
            )


# For CFHT_1044 survey
W1_input = '/cosma5/data/durham/dc-koon1/PAUS/PAUS_W1_1044_CFHT_complete_i.csv'
process_survey(W1_input, W1_1044_selection_LFcal, dp=0.01, n_points=1000000)

# For W3_1045 survey 
W3_input = '/cosma5/data/durham/dc-koon1/PAUS/PAUS_W3_1045_CFHT_complete_i.csv'
process_survey(W3_input, W3_1045_selection_LFcal, dp=0.01, n_points=1000000)


def W1_1044_selection_GalaxyOnly(catalog):
    """
    Selection function for CFHT_1044 survey.
    Modify the criteria as needed.
    """
    return ((catalog.star_flag == 0) & (catalog['mask'] <= 1)
    )

def W3_1045_selection_GalaxyOnly(catalog):
    """
    Example selection function for W3_1045 survey.
    Adjust the criteria for W3 as needed.
    """
    return ((catalog.star_flag == 0) & (catalog['mask'] <= 1)
            )

# For CFHT_1044 survey
W1_input = '/cosma5/data/durham/dc-koon1/PAUS/PAUS_W1_1044_CFHT_complete_i.csv'
process_survey(W1_input, W1_1044_selection_GalaxyOnly, dp=0.01, n_points=1000000)

# For W3_1045 survey 
W3_input = '/cosma5/data/durham/dc-koon1/PAUS/PAUS_W3_1045_CFHT_complete_i.csv'
process_survey(W3_input, W3_1045_selection_GalaxyOnly, dp=0.01, n_points=1000000)
