import cv2
import numpy as np
from sklearn.cluster import KMeans


def parse_yolo_labels(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = list(map(float, line.split()))

            cls, x1, x2, y1, y2 = parts
            if cls==1 or cls==2 or cls==5:
                bounding_boxes.append([cls, x1, x2, y1, y2])
    return bounding_boxes


def calculate_color_distance(color1, color2):
    """
    Calculate color distance using a combination of RGB and HSV spaces
    """
    # Convert to HSV
    hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2HSV)[0][0]

    # Calculate distances in both spaces
    rgb_dist = np.sqrt(np.sum((color1 - color2) ** 2))

    # Weighted HSV distance (giving more importance to hue)
    h_dist = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0])) / 90.0
    s_dist = abs(hsv1[1] - hsv2[1]) / 255.0
    v_dist = abs(hsv1[2] - hsv2[2]) / 255.0

    hsv_dist = np.sqrt(4 * h_dist ** 2 + s_dist ** 2 + v_dist ** 2)

    # Combine distances with weights
    return 0.5 * rgb_dist + 0.5 * hsv_dist * 255


def apply_masking(crop):
    # Refined ground color masks
    # Greens (for field) and browns (for dirt)
    # Convert to HSV for better color separation
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv_crop, lower_green, upper_green)

    # Create dynamic threshold for field color
    field_sample = hsv_crop[hsv_crop.shape[0] // 2:, :, :]  # Sample from bottom half
    field_hue = np.median(field_sample[:, :, 0])
    field_lower = np.array([max(0, field_hue - 10), 30, 30])
    field_upper = np.array([min(180, field_hue + 10), 255, 255])
    field_mask = cv2.inRange(hsv_crop, field_lower, field_upper)

    # Combine field masks
    ground_mask = cv2.bitwise_or(green_mask, field_mask)
    player_mask = cv2.bitwise_not(ground_mask)

    # Apply color range for jersey colors
    lower_uniform = np.array([0, 30, 40])
    upper_uniform = np.array([180, 255, 255])

    color_mask = cv2.inRange(hsv_crop, lower_uniform, upper_uniform)

    # Combine masks
    final_mask = cv2.bitwise_and(player_mask, color_mask)

    # Clean up mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask to processed image
    masked_crop = cv2.bitwise_and(crop, crop, mask=final_mask)

    pixels = masked_crop.reshape((-1, 3))
    pixels = pixels[np.any(pixels != 0, axis=1)]
    return pixels


def extract_dominant_colors(crop, n_colors=2):
    pixels = apply_masking(crop)
    if len(pixels) < 10:  # If still too few pixels
        # Take center portion of crop
        h, w = crop.shape[:2]
        center_crop = crop[h // 4:3 * h // 4, w // 4:3 * w // 4]
        pixels = center_crop.reshape(-1, 3)

    # Use KMeans with increased max iterations for better convergence
    kmeans = KMeans(n_clusters=n_colors, random_state=22, n_init=10, max_iter=300)

    kmeans.fit(pixels)

    # Process clusters
    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    centers = kmeans.cluster_centers_

    # Create color-count pairs
    color_counts = list(zip(centers, counts))
    color_counts.sort(key=lambda x: x[1], reverse=True)

    # Filter colors with refined criteria
    filtered_colors = []
    for color, count in color_counts:
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]

        # Refined color filtering criteria
        if (hsv_color[1] >= 30 and  # Minimum saturation
                20 <= hsv_color[2] <= 240 and  # Value range
                count / len(pixels) > 0.1):  # Minimum area threshold

            # Skip if it matches field color
            if not (35 <= hsv_color[0] <= 85 and hsv_color[1] > 40):
                filtered_colors.append(tuple(map(int, color)))

        if len(filtered_colors) == 2:
            break

    # Pad with black if needed
    while len(filtered_colors) < 2:
        filtered_colors.append((0, 0, 0))

    return filtered_colors


def extract_player_colors(image, bounding_boxes):
    """
    Extract colors with enhanced visualization.
    """
    player_colors = []

    for i, box in enumerate(bounding_boxes):
        cls, x1, x2, y1, y2 = box

        # Crop the region
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue

        # Process and extract colors

        dominant_colors = extract_dominant_colors(crop)
        player_colors.append((dominant_colors, box))

        # Debug visualization
        small_crop = cv2.resize(crop, (200, 200))

        # Create color patches
        color_patch = np.zeros((40, 400, 3), dtype=np.uint8)
        color_patch[:, :200] = dominant_colors[0]
        if (len(dominant_colors) > 1):
            color_patch[:, 200:] = dominant_colors[1]

        # Create comparison view

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(small_crop, 'Original', (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(color_patch, 'Colors', (10, 220), font, 0.5, (255, 255, 255), 1)

        # cv2.imshow(f"Player {i} Analysis", small_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return player_colors


def classify_players_by_color(player_colors, n_teams=2, referee_threshold=0.01):
    """
    Classify players using both dominant colors, with referee detection.

    Args:
        player_colors: List of color pairs and bounding boxes
        n_teams: Number of teams (default 2)
        referee_threshold: Distance threshold to identify referees
    """
    # Extract primary colors for clustering
    primary_colors = np.array([color_pair[0][0] for color_pair in player_colors])
    secondary_colors = np.array([color_pair[0][1] for color_pair in player_colors])

    # Combine color features
    color_features = np.concatenate([primary_colors, secondary_colors], axis=1)

    # Normalize features
    color_features = color_features / 255.0

    # Perform clustering for teams
    kmeans = KMeans(n_clusters=n_teams, random_state=20, n_init=10)
    labels = kmeans.fit_predict(color_features)

    # Get cluster centers
    team_centers = kmeans.cluster_centers_

    # Calculate distances to cluster centers for each player
    distances = np.zeros((len(color_features), n_teams))
    for i in range(n_teams):
        distances[:, i] = np.linalg.norm(color_features - team_centers[i], axis=1)

    # Find minimum distance to any team cluster
    min_distances = np.min(distances, axis=1)

    # Mark players as referees if they're too far from both team clusters
    referee_mask = min_distances > referee_threshold

    # Create final labels: teams are 0 and 1, referee is 2
    final_labels = labels.copy()
    final_labels[referee_mask] = 2

    return final_labels


def calculate_clustering_accuracy(predicted_labels, true_labels):
    """
    Calculate clustering accuracy considering possible label permutations and referees.

    Args:
        predicted_labels (numpy.ndarray): Labels from clustering (0, 1, or 2 for referee)
        true_labels (numpy.ndarray): Ground truth labels from YOLO file
    """
    # Handle only team labels for accuracy calculation
    team_mask = predicted_labels != 2
    if not any(team_mask):
        return 0.0, {'true_positive': 0, 'true_negative': 0,
                     'false_positive': 0, 'false_negative': 0}

    team_predictions = predicted_labels[team_mask]
    team_true_labels = true_labels[team_mask] - 1  # Convert from 1,2 to 0,1

    # Try both possible label mappings
    accuracy1 = np.mean(team_predictions == team_true_labels)
    accuracy2 = np.mean(team_predictions == (1 - team_true_labels))

    best_accuracy = max(accuracy1, accuracy2)

    # Use the best mapping for confusion matrix
    if accuracy1 >= accuracy2:
        final_predictions = team_predictions
    else:
        final_predictions = 1 - team_predictions

    # Calculate confusion matrix
    confusion_matrix = {
        'true_positive': np.sum((final_predictions == 1) & (team_true_labels == 1)),
        'true_negative': np.sum((final_predictions == 0) & (team_true_labels == 0)),
        'false_positive': np.sum((final_predictions == 1) & (team_true_labels == 0)),
        'false_negative': np.sum((final_predictions == 0) & (team_true_labels == 1))
    }

    return best_accuracy, confusion_matrix


def main():
    image_path = '15.png'
    labels_path = '15.txt'

    # Load and process image
    original_image = cv2.imread(image_path)

    # Parse bounding boxes and extract colors
    bounding_boxes = parse_yolo_labels(labels_path)
    player_colors = extract_player_colors(original_image, bounding_boxes)

    # Classify players with referee detection
    team_labels = classify_players_by_color(player_colors, referee_threshold=0.3)
    true_labels = np.array([box[0] for box in bounding_boxes])

    # Visualize results
    vis_image = original_image.copy()
    TEAM1_COLOR = (0, 0, 255)  # Red in BGR
    TEAM2_COLOR = (255, 0, 0)  # Blue in BGR
    REFEREE_COLOR = (0, 255, 0)  # Green in BGR

    for (colors, box), team_label in zip(player_colors, team_labels):
        cls, x1, x2, y1, y2 = box

        # Choose color based on team label
        if team_label == 2:  # Referee
            box_color = REFEREE_COLOR
            label_text = "REF"
        else:
            box_color = TEAM1_COLOR if team_label == 0 else TEAM2_COLOR
            label_text = f"T {team_label + 1}"

        # Draw bounding box
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)),
                      box_color, 2)

        # Add color patches for visualization
        patch_width = 30
        patch_height = 15
        y_offset = int(y1) - patch_height - 5

        # Draw patches for both colors
        for i, color in enumerate(colors):
            patch_x = int(x1) + (i * patch_width)
            if y_offset > 0:
                cv2.rectangle(vis_image,
                              (patch_x, y_offset),
                              (patch_x + patch_width, y_offset + patch_height),
                              color, -1)

        # Add label
        cv2.putText(vis_image, label_text,
                    (int(x1), int(y1) - patch_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    accuracy, confusion_matrix = calculate_clustering_accuracy(team_labels, true_labels)

    print("\nClustering Performance Metrics:")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Display results
    cv2.imshow('Team Classification with Colors', vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


