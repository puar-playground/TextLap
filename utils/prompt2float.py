from collections import Counter
import json
import random
from scipy.cluster.hierarchy import fcluster, linkage
import numpy as np


def prompt_float(descriptions, labels, bounding_boxes, size):

    (w, h) = size
    if len(descriptions) != len(labels) or len(descriptions) != len(bounding_boxes):
        raise ValueError("The number of descriptions, labels, and bounding boxes must be the same.")
    
    # Count occurrences of each label
    label_count = Counter(labels)
    
    # Track the current index for each label
    label_index = {label: 1 for label in label_count}
    
    template = []
    modified_labels = []
    
    for caption, label, bounding_box in zip(descriptions, labels, bounding_boxes):
        # Append the index to the label if it occurs more than once
        if label_count[label] > 1:
            element_name = f"{label}{label_index[label]}"
            label_index[label] += 1
        else:
            element_name = label
        
        # Calculate width and height from bounding box
        x1, y1, x2, y2 = bounding_box
        width = round((x2 - x1) / w, 3)
        height = round((y2 - y1) / h, 3)

        caption = caption.replace('\n', ' ')
        template.append({'element_name': element_name, 'caption': caption, 'width': width, 'height': height})
        
        modified_labels.append(element_name)

    template_string = json.dumps(template, indent=4)
    
    return template_string, modified_labels



def response_float(element_names, coordinates, size):
    (w, h) = size
    if len(element_names) != len(coordinates):
        raise ValueError("The number of labels and coordinates must be the same.")
    
    template = dict()
    
    for element_name, (left, top, right, bottom) in zip(element_names, coordinates):

        width = right - left
        height = bottom - top
        
        template[element_name] = [round(left/w, 3), round(top/h, 3), round(right/w, 3), round(bottom/h, 3)]
    
    template_string = json.dumps(template)
    
    return template_string



if __name__=="__main__":
    
    # Example usage
    descriptions = [
        "The background is a dark, almost black canvas with a few yellow elements that resemble lines and shapes, possibly representing a minimalist design or abstract art.",
        "The image shows a black and white photograph of a lion statue with a shield, positioned on a pedestal in front of a building with a decorative archway and a wall sconce.",
        "A text that says: \"Urban\\nLegends\"",
        "The image is a solid yellow color with no discernible features or text.",
        "A text that says: \"SCARIEST\""
    ]
    
    labels = ["background", "imageElement", "textElement", "svgElement", "textElement"]
    
    bounding_boxes = [
        (0, 0, 1000, 1000),
        (100, 150, 400, 450),
        (200, 200, 500, 300),
        (300, 300, 600, 600),
        (400, 400, 700, 450)
    ]
    
    css_template, modified_labels = prompt_css(descriptions, labels, bounding_boxes)
    print(css_template)
    print(modified_labels)





    
    
