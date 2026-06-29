import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

# ==========================================
# 4-f Optical Correlator — Fingerprint Matcher
# ==========================================

# ==========================================
# 1. Utilities
# ==========================================
def create_2d_window(h, w):
    """
    2D Hamming window to suppress boundary artifacts before FFT.
    Fades image edges smoothly to zero so the FFT doesn't see hard edges.
    """
    return np.outer(np.hamming(h), np.hamming(w))

def compute_fourier_plane(image):
    """
    Simulates Lens 1 of the 4-f system: spatial image -> Fourier plane.
    Returns the full complex spectrum (not just magnitude), DC-centred.
    """
    h, w = image.shape
    window = create_2d_window(h, w)
    field = (image.astype(float) / 255.0) * window
    return np.fft.fftshift(np.fft.fft2(field))

def compute_fourier_mask(db_image, threshold_percentile=95):
    """
    Builds the spatial filter mask placed at the Fourier plane.
    """
    spectrum = compute_fourier_plane(db_image)
    mag = np.abs(spectrum)
    mag_log = np.log1p(mag)

    threshold = np.percentile(mag_log, threshold_percentile)

    # Transmittance: 0 where DB fingerprint is strong, 1 elsewhere
    mask = (mag_log < threshold).astype(float)
    return mask

# ==========================================
# 2. The 4-f Correlator
# ==========================================
def fourier_plane_correlate(query_image, fourier_plane_mask):
    """
    Simulates the full 4-f optical correlator for one query/mask pair.
    """
    # --- Stage 1: Lens 1 — forward Fourier transform ---
    query_spectrum = compute_fourier_plane(query_image)

    # --- Stage 2: Apply mask at Fourier plane ---
    filtered_spectrum = query_spectrum * fourier_plane_mask

    # --- Stage 3: Lens 2 — inverse Fourier transform ---
    output_field = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))
    output_intensity = np.abs(output_field)

    # --- Stage 4: Measure output intensity ---
    mean_intensity = float(np.mean(output_intensity))

    # Invert: low residual intensity = good match = high score
    score = 1.0 - min(mean_intensity * 4.0, 1.0)
    return score

# ==========================================
# 3. Database & Identification System
# ==========================================
class FourFCorrelatorDatabase:
    def __init__(self, match_threshold=0.85, gap_threshold=0.01,
                 fourier_mask_percentile=95, debug=False):
        self.database = {}
        self.match_threshold = match_threshold
        self.gap_threshold = gap_threshold
        self.fourier_mask_percentile = fourier_mask_percentile
        self.debug = debug

    def enroll(self, person_id, image):
        mask = compute_fourier_mask(image, self.fourier_mask_percentile)
        self.database[person_id] = mask
        print(f"  Enrolled '{person_id}' — Fourier plane mask computed.")

    def identify(self, query_image):
        """
        Modified to always return the best matching ID, even if it fails 
        the threshold, so we can plot the failure.
        Returns: (status_string, score, best_matched_id)
        """
        scores = {
            pid: fourier_plane_correlate(query_image, mask)
            for pid, mask in self.database.items()
        }

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if self.debug:
            print("  Score distribution:")
            for pid, score in sorted_scores:
                print(f"    {pid:<20} {score:.6f}")

        best_id, best_score = sorted_scores[0]
        status = best_id

        # Reject if best score doesn't clear the threshold
        if best_score < self.match_threshold:
            status = "No Match Found"
        # Ambiguous only if two high scores are too close together
        elif len(sorted_scores) > 1:
            second_score = sorted_scores[1][1]
            if (best_score - second_score) < self.gap_threshold:
                status = "Ambiguous"

        return status, best_score, best_id

# ==========================================
# 4. Helpers & Visualization
# ==========================================
def load_images_from_folder(folder, resolution):
    valid_exts = ('*.png', '*.jpg', '*.jpeg', '*.tif')
    paths = []
    for ext in valid_exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths.sort()

    loaded = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, resolution)
            loaded.append((path, img))
    return loaded

def visualize_4f_stages(query_image, db_mask, title="4-f Correlator Optical Stages"):
    query_spectrum = compute_fourier_plane(query_image)
    query_mag_display = np.log1p(np.abs(query_spectrum))
    filtered_spectrum = query_spectrum * db_mask
    output_field = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))
    output_intensity = np.abs(output_field)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(query_image, cmap='gray')
    ax[0].set_title("1. Input Fingerprint")
    ax[0].axis('off')

    ax[1].imshow(query_mag_display, cmap='inferno')
    ax[1].set_title("2. Input Fourier Spectrum")
    ax[1].axis('off')

    ax[2].imshow(db_mask, cmap='gray')
    ax[2].set_title("3. Database Filter Mask")
    ax[2].axis('off')

    ax[3].imshow(output_intensity, cmap='inferno')
    ax[3].set_title("4. Detection Plane (Output)")
    ax[3].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def run_evaluation(db, image_list, threshold, label="", plot_results=True):
    if label:
        print(f"\n--- {label} ---")

    print(f"==========================================================================")
    print(f"{'TEST IMAGE FILE':<30} | {'PREDICTED ID':<20} | {'SCORE':<7} | {'RESULT'}")
    print(f"==========================================================================")

    success_count = 0
    ambiguous_count = 0
    total_count = len(image_list)

    for path, img in image_list:
        filename = os.path.basename(path)
        if db.debug:
            print(f"\n  Query: {filename}")
            
        # Get the match status, score, AND the highest scoring mask ID
        match_status, score, best_id = db.identify(img)

        if match_status == "Ambiguous":
            status_icon = "AMBIGUOUS"
            ambiguous_count += 1
        elif match_status == "No Match Found":
            status_icon = "FAILED"
        else:
            status_icon = "MATCH"
            success_count += 1

        print(f"{filename:<30} | {match_status:<20} | {score:.6f} | {status_icon}")

        # Pop up the visualization for this specific test
        if plot_results:
            top_mask = db.database[best_id]
            plot_title = f"Query: {filename} | Top Candidate: {best_id} | Score: {score:.4f} | {status_icon}"
            visualize_4f_stages(img, top_mask, title=plot_title)

    print(f"==========================================================================")
    print(f"Summary: {success_count} matched, {ambiguous_count} ambiguous, "
          f"{total_count - success_count - ambiguous_count} failed — out of {total_count} total.")

# ==========================================
# 5. Automated Evaluation Loop
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    DB_FOLDER    = "database"        
    NOISY_FOLDER = "noisy_database"  
    RESOLUTION   = (256, 256)

    THRESHOLD               = 0.85
    GAP_THRESHOLD           = 0.01   
    FOURIER_MASK_PERCENTILE = 95     

    DEBUG = False # Set to False by default so the terminal output is clean

    db = FourFCorrelatorDatabase(
        match_threshold=THRESHOLD,
        gap_threshold=GAP_THRESHOLD,
        fourier_mask_percentile=FOURIER_MASK_PERCENTILE,
        debug=DEBUG
    )

    # --- 1. Enroll Clean Database ---
    print("Enrolling fingerprints into 4-f correlator database...")
    db_images = load_images_from_folder(DB_FOLDER, RESOLUTION)
    for path, img in db_images:
        person_id = os.path.splitext(os.path.basename(path))[0]
        db.enroll(person_id, img)
    print(f"\nSuccessfully enrolled {len(db.database)} Fourier plane masks.\n")

    # --- 2. Run 4-f Correlator on Noisy Images ---
    noisy_images = load_images_from_folder(NOISY_FOLDER, RESOLUTION)
    if noisy_images:
        # We pass plot_results=True to trigger the popup for every image
        run_evaluation(db, noisy_images, THRESHOLD, label="Noisy Spatial Images", plot_results=True)
    else:
        print(f"No images found in '{NOISY_FOLDER}', skipping.")

    print(f"\nThreshold: {THRESHOLD}  |  Gap: {GAP_THRESHOLD}  |  "
          f"Mask percentile: {FOURIER_MASK_PERCENTILE}  |  Debug: {DEBUG}")