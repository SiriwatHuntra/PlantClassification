import cv2
import numpy as np
from skimage import feature
import pywt
from scipy.fftpack import fft
import networkx as nx
from scipy.stats import entropy
from scipy.stats import linregress
from numba import jit
from sklearn.preprocessing import StandardScaler

# Utility Functions
def preprocess_image(image, to_gray=True):
    """Optimized image preprocessing with Gaussian Blur and adaptive thresholding."""
    
    if to_gray and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.GaussianBlur(image, (5, 5), 0)  # Smoother blurring before thresholding

    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)  # Reduced iterations

    return binary

def compute_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Efficient contour filtering and selection
    valid_contours = [c for c in contours if 500 < cv2.contourArea(c) < binary_image.size * 0.9]
    
    if not valid_contours:
        return None, None

    largest_contour = max(valid_contours, key=cv2.contourArea)
    approx_contour = cv2.approxPolyDP(largest_contour, 0.015 * cv2.arcLength(largest_contour, True), True)

    return largest_contour, approx_contour


# Chain Code and Fourier Descriptors
def extract_chain_code_features(contour):
    if contour.ndim == 3:
        contour = contour[:, 0, :]

    if len(contour) < 2:
        return {f"direction_{i}": 0 for i in range(8)} | {"smoothness": 0, "irregularity": 0, "same_pattern_score": 0}

    # Compute directional differences
    dx_dy = np.sign(np.diff(contour, axis=0))  # (N-1, 2)

    # Map (dx, dy) pairs to chain code values
    direction_map = {
        (-1, 0): 0, (-1, 1): 1, (0, 1): 2, (1, 1): 3,
        (1, 0): 4, (1, -1): 5, (0, -1): 6, (-1, -1): 7
    }

    chain_code = np.array([direction_map.get(tuple(d), -1) for d in dx_dy])
    chain_code = chain_code[chain_code != -1]  # Remove invalid values

    # Compute direction distribution (histogram)
    direction_distribution = np.histogram(chain_code, bins=8, range=(0, 8))[0]

    # Compute smoothness (Average of first-order differences)
    if len(chain_code) > 1:
        first_order_diff = np.abs(np.diff(chain_code) % 8)
        smoothness = np.mean(first_order_diff)
    else:
        smoothness = 0

    # Compute same pattern score (frequency of repeating sequences)
    def repeating_subsequences(chain, min_length=2, max_length=10):
        """
        Identify repeating subsequences in chain code to determine pattern consistency.
        Higher score means more repetition.
        """
        n = len(chain)
        pattern_count = 0

        for length in range(min_length, max_length + 1):
            seen = {}
            for i in range(n - length + 1):
                sub = tuple(chain[i : i + length])
                if sub in seen:
                    seen[sub] += 1
                else:
                    seen[sub] = 1
            pattern_count += sum(v - 1 for v in seen.values() if v > 1)

        return pattern_count / max(1, len(chain))  # Normalize by chain length

    same_pattern_score = repeating_subsequences(chain_code)

    return {
        "smoothness": round(smoothness, 6),
        "same_pattern_score": round(same_pattern_score, 6)
    }


import numpy as np
import cv2

def extract_shape_corners(contour):
    """
    Extracts shape descriptors from approx contour:
    - edge count
    - sharp/obtuse angles
    - average angle
    - edge length statistics
    - compactness
    - convexity
    """
    if contour is None or len(contour) < 4:
        return {
            "num_edges": 0,
            "num_sharp_edges": 0,
            "num_obtuse_edges": 0,
            "avg_angle": 0,
            "edge_len_mean": 0,
            "edge_len_std": 0,
            "compactness": 0,
            "convexity_ratio": 0,
        }

    contour_points = np.squeeze(contour, axis=1)  # (N,2)

    if len(contour_points) < 4:
        return {
            "num_edges": len(contour_points),
            "num_sharp_edges": 0,
            "num_obtuse_edges": 0,
            "avg_angle": 0,
            "edge_len_mean": 0,
            "edge_len_std": 0,
            "compactness": 0,
            "convexity_ratio": 0,
        }

    prev_points = np.roll(contour_points, shift=1, axis=0)
    next_points = np.roll(contour_points, shift=-1, axis=0)

    vectors1 = prev_points - contour_points
    vectors2 = next_points - contour_points

    norms1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(vectors2, axis=1, keepdims=True)

    unit_vectors1 = np.divide(vectors1, norms1, where=norms1 != 0)
    unit_vectors2 = np.divide(vectors2, norms2, where=norms2 != 0)

    dot_products = np.sum(unit_vectors1 * unit_vectors2, axis=1)
    angles = np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0)))

    num_sharp = np.sum(angles < 90)
    num_obtuse = np.sum(angles >= 90)
    avg_angle = np.mean(angles) if angles.size > 0 else 0

    # Edge lengths
    edge_lengths = np.linalg.norm(np.diff(np.vstack([contour_points, contour_points[0]]), axis=0), axis=1)
    edge_len_mean = np.mean(edge_lengths)
    edge_len_std = np.std(edge_lengths)

    # Compactness: Perimeter^2 / (4π * Area)
    area = cv2.contourArea(contour_points)
    perimeter = cv2.arcLength(contour_points, True)
    compactness = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0

    # Convexity ratio: Perimeter / Convex hull perimeter
    hull = cv2.convexHull(contour_points)
    hull_perimeter = cv2.arcLength(hull, True)
    convexity_ratio = perimeter / hull_perimeter if hull_perimeter > 0 else 0

    return {
        "num_edges": len(contour_points),
        "num_sharp_edges": int(num_sharp),
        "num_obtuse_edges": int(num_obtuse),
        "avg_angle": round(avg_angle, 6),
        "edge_len_mean": round(edge_len_mean, 6),
        "edge_len_std": round(edge_len_std, 6),
        "compactness": round(compactness, 6),
        "convexity_ratio": round(convexity_ratio, 6),
    }



# Rim Pattern Analysis
def analyze_rim_pattern(contour):
    """
    Analyzes the rim pattern of a given contour by computing statistical properties
    of the differences between consecutive contour points.

    :param contour: Contour points (numpy array of shape Nx1x2)
    :return: Dictionary with rim pattern statistics 
    """
    if contour is None or len(contour) < 2:
        raise ValueError("Invalid contour: must have at least two points.")

    # Convert contour to NumPy array for efficient computation
    contour_points = contour[:, 0, :]  # Extract Nx2 shape

    # Compute difference vectors efficiently
    diff_vectors = np.linalg.norm(np.diff(contour_points, axis=0, append=contour_points[:1]), axis=1)

    # Compute statistical features
    std_dev = np.std(diff_vectors)
    mean_diff = np.mean(diff_vectors)
    diff_entropy = entropy(diff_vectors + 1e-10)  # Avoid log(0) issues

    # Return results with 6-digit precision
    return {
        "rim_pattern_std_dev": round(std_dev, 6),
        "rim_pattern_mean": round(mean_diff, 6),
        "rim_pattern_entropy": round(diff_entropy, 6)
    }

def extract_texture_features(image):
    """
    Extracts Gray-Level Co-occurrence Matrix (GLCM) texture features efficiently,
    ensuring accuracy and returning values rounded to 6 decimal places.

    :param image: Input image (numpy array)
    :return: Dictionary with GLCM texture features
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input: image is empty or None")

    # Convert to grayscale efficiently
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define GLCM parameters
    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Compute Gray-Level Co-occurrence Matrix (GLCM)
    glcm = feature.graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    # Extract texture properties efficiently
    props = ['contrast', 'energy', 'homogeneity', 'correlation']
    features = {prop: round(feature.graycoprops(glcm, prop).mean(), 6) for prop in props}

    return features

def extract_fourier_Descriptor(contour, min_harmonic=50, max_harmonic=60):
    """
    Extracts magnitude-only Fourier Descriptors (FD) for shape analysis.
    
    :param contour: numpy array of shape (N, 1, 2) or (N, 2), representing the contour points.
    :param min_harmonic: Minimum frequency index (inclusive)
    :param max_harmonic: Maximum frequency index (inclusive)
    :return: Dictionary with magnitude values of Fourier coefficients.
    """
    if contour.ndim == 3:
        contour = contour[:, 0, :]  # Ensure shape is (N, 2)

    if len(contour) < 3 or max_harmonic < min_harmonic:
        return {f"efc_magnitude_{i}": 0.0 for i in range(min_harmonic, max_harmonic + 1)}

    # Compute cyclic contour differences
    dxy = np.diff(np.vstack([contour, contour[:1]]), axis=0)

    # Convert to complex representation
    complex_signal = dxy[:, 0] + 1j * dxy[:, 1]

    # Compute Fourier descriptors
    descriptors = fft(complex_signal)

    # Normalize by the magnitude of the first harmonic (scale invariance)
    normalization_factor = np.abs(descriptors[1]) if not np.isclose(descriptors[1], 0) else 1
    normalized_descriptors = descriptors / normalization_factor

    # Take magnitude from selected harmonic range
    magnitude_features = np.abs(normalized_descriptors[min_harmonic:max_harmonic + 1])

    # Return as dictionary
    return {f"efc_magnitude_{i}": float(magnitude_features[i - min_harmonic]) for i in range(min_harmonic, max_harmonic + 1)}


def extract_vein_features(binary_image):
    binary_image = np.uint8(binary_image)
    skeleton = cv2.ximgproc.thinning(binary_image)
    # cv2.imshow("Skel", skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    vein_density = np.count_nonzero(skeleton) / skeleton.size

    yx_coords = np.column_stack(np.where(skeleton > 0))
    node_set = set(map(tuple, yx_coords))

    G = nx.Graph()
    G.add_nodes_from(node_set)

    neighbor_offsets = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]])
    
    for xy in yx_coords:
        neighbors = xy + neighbor_offsets
        valid_neighbors = [tuple(n) for n in neighbors if tuple(n) in node_set]
        G.add_edges_from((tuple(xy), neighbor) for neighbor in valid_neighbors)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degree_sequence = [d for _, d in G.degree()]
    avg_degree = np.mean(degree_sequence) if degree_sequence else 0.0

    num_cycles = len(list(nx.cycle_basis(G))) if num_nodes > 1 else 0

    return {
        "vein_density": vein_density,
        "vein_loop_count": num_cycles,
        "vein_graph_nodes": num_nodes,
        "vein_graph_edges": num_edges,
        "vein_avg_branch_degree": avg_degree
    }

def compute_fractal_dimension(image):
    """
    Computes the fractal dimension of an image using an optimized Box-Counting method.
    """

    # Efficient skeletonization using thinning (Guo-Hall method for better accuracy)
    skeleton = cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    
    # cv2.imshow("Skeletonize", skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    @jit(nopython=True)
    def box_count(image, size):
        """Count number of boxes of size `size` containing nonzero pixels."""
        w, h = image.shape
        count = 0
        for i in range(0, w, size):
            for j in range(0, h, size):
                if np.any(image[i:i + size, j:j + size]):
                    count += 1
        return count

    # Generate logarithmic scale box sizes (avoid redundant sizes)
    sizes = np.unique(np.logspace(1, np.log2(min(image.shape)), num=10, base=2, dtype=int))

    # Perform parallelized box counting
    counts = np.array([box_count(skeleton, size) for size in sizes], dtype=float)
    counts[counts == 0] = np.finfo(float).eps  # Avoid log(0) issue

    # Compute linear regression using SciPy (more robust than polyfit)
    log_sizes = np.log2(sizes)
    log_counts = np.log2(counts)
    slope, intercept, _, _, _ = linregress(log_sizes, log_counts)

    return {"fractal_dimension": -slope}

def extract_wavelet_texture(image):
    """
    Efficiently extracts multi-scale texture features using Discrete Wavelet Transform (DWT),
    applies feature standardization, and rounds results to 6 decimal places.
    
    :param image: Input color image (numpy array)
    :return: Dictionary containing standardized and rounded wavelet texture features
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input: image is empty or None")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grey", gray)
    # cv2.waitKey(0)
    # cv2.destroyWindow()
    # Normalize image to range [0,1] for numerical stability
    gray = gray.astype(np.float32) / 255.0

    # Perform 2-level Discrete Wavelet Transform (DWT) with 'db2' wavelet
    coeffs = pywt.wavedec2(gray, 'db2', level=2)

    # Extract wavelet features
    feature_list = []
    feature_names = []

    for i, (cH, cV, cD) in enumerate(coeffs[1:], start=1):  # Ignore approximation coefficients
        coeff_matrices = {'H': cH, 'V': cV, 'D': cD}

        for direction, coeff in coeff_matrices.items():
            abs_coeff = np.abs(coeff)  # Avoid negative values affecting calculations

            # Compute features
            mean_val = np.mean(abs_coeff)
            std_val = np.std(abs_coeff)
            energy_val = np.sum(abs_coeff ** 2) / abs_coeff.size  # Energy measure
            entropy_val = entropy(abs_coeff.ravel() + 1e-10)  # Avoid log(0) issues

            # Store values
            feature_list.extend([mean_val, std_val, energy_val, entropy_val])
            feature_names.extend([
                f"wavelet_{direction}_mean_{i}",
                f"wavelet_{direction}_std_{i}",
                f"wavelet_{direction}_energy_{i}",
                f"wavelet_{direction}_entropy_{i}"
            ])

    # Standardization (Z-score normalization)
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(np.array(feature_list).reshape(-1, 1)).flatten()

    # Create a dictionary with rounded values
    features = {name: round(value, 6) for name, value in zip(feature_names, standardized_features)}

    return features


def compute_curvature_analysis(contour):
    """
    Computes the average and maximum curvature of a given contour.
    
    Parameters:
        contour (numpy.ndarray): A Nx1x2 or Nx2 array representing contour points.
    
    Returns:
        dict: Contains 'avg_curvature' and 'max_curvature' values.
    """
    contour_points = np.squeeze(contour)  # Removes single-dimensional entries

    if contour_points.shape[0] < 5:  # Ensure sufficient points for curvature analysis
        return {"avg_curvature": 0.0, "max_curvature": 0.0}

    # Compute first and second derivatives
    dx_dt, dy_dt = np.gradient(contour_points[:, 0]), np.gradient(contour_points[:, 1])
    d2x_dt2, d2y_dt2 = np.gradient(dx_dt), np.gradient(dy_dt)

    # Compute denominator for curvature equation
    speed_sq = dx_dt**2 + dy_dt**2  # Squared speed
    curvature_denom = np.maximum(speed_sq ** 1.5, np.finfo(float).eps)  # Avoid division by zero

    # Compute curvature using optimized vectorized operations
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / curvature_denom

    return {
        "avg_curvature": np.mean(curvature) if curvature.size else 0.0,
        "max_curvature": np.max(curvature) if curvature.size else 0.0
    }

# Feature Extraction Pipeline
def extract_features(image):
    binary = preprocess_image(image)
    contour, lowpoly = compute_contour(binary)

    features = {
        **extract_chain_code_features(lowpoly),
        **extract_fourier_Descriptor(contour),
        **extract_shape_corners(lowpoly),
        **analyze_rim_pattern(contour),
        **extract_texture_features(image),
        **extract_wavelet_texture(image),
        **extract_vein_features(binary),
        **compute_fractal_dimension(binary),
        **compute_curvature_analysis(contour),
    }

    return features