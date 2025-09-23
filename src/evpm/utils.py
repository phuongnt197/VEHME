import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import xml.etree.ElementTree as ET
from io import BytesIO
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np

MAX_PADDING = 20
MAX_ROTATE = 30

def render(traces) -> BytesIO:
    # Set the figure size to (224, 224)
    canvas_height = 224
    fig, ax = plt.subplots()
    # Set plot size
    num_points = sum([len(trace) for trace in traces])

    

    x_min = min([min([x for x, y in trace]) for trace in traces])
    y_min = min([min([y for x, y in trace]) for trace in traces])

    x_max = max([max([x for x, y in trace]) for trace in traces])
    y_max = max([max([y for x, y in trace]) for trace in traces])

    width = x_max - x_min
    height = y_max - y_min

    scale_height = canvas_height / height
    aspect_ratio = width / height
    canvas_width = int(canvas_height * aspect_ratio)
    fig.set_size_inches(canvas_width / 100, canvas_height / 100)
    scale_width = canvas_width / width

    traces = [
        [((x - x_min) * scale_width, (y - y_min) * scale_height) for x, y in trace]
        for trace in traces
    ]

    for trace in traces:
        x, y = zip(*trace)
        ax.plot(x, y, color='black', linewidth=1.5)  # Increase the linewidth to 2
    ax.invert_yaxis()
    ax.axis('off')
    
    # Save plot to an in-memory buffer with transparent background
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)

    return buf

def get_traces_from_inkml(inkml_file):
    tree = ET.parse(inkml_file)
    root = tree.getroot()
    traces = []
    for child in root:
        if child.tag == "{http://www.w3.org/2003/InkML}trace":
            trace = [coord.split() for coord in child.text.strip().split(",")]
            trace = [(float(x), float(y)) for x, y in trace]
            traces.append(trace)
    return traces

def rotate_point(x, y, angle_deg):
    theta = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    original = np.array([x, y])
    rotated = rotation_matrix @ original
    return rotated[0], rotated[1]

MAX_PADDING = 20
MAX_ROTATE = 30
def create_canvas(hmes):

    ink_traces = [get_traces_from_inkml(inkml_file) for inkml_file in hmes]
    ink_imgs = [Image.open(render(traces)) for traces in ink_traces]

    max_width = max([img.width for img in ink_imgs])
    max_height = max([img.height for img in ink_imgs])

    # Create a blank canvas
    canvas_width, canvas_height = 2 * max_width + MAX_PADDING, (np.sqrt(max_height ** 2 + max_width ** 2) + MAX_PADDING) * (len(hmes))

    canvas = Image.new('RGBA', (int(canvas_width), int(canvas_height)), (255, 255, 255, 255))

    x_offset = 0
    y_offset = 0

    list_of_bboxes = []

    for img in ink_imgs:
        # Resize the image to (whatever to keep the aspect ratio, 224)
        # img = img.resize((int(img.width * (224 / img.height)), 224))
        
        # Apply a small random angle perturbation
        angle = random.uniform(-MAX_ROTATE, MAX_ROTATE)
        rotated_img = img.rotate(angle, expand=True)

        x_offset = random.randint(0, canvas_width - rotated_img.width)

        delta_x = (rotated_img.size[0] - img.size[0])
        delta_y = (rotated_img.size[1] - img.size[1])

        # Paste the rotated image onto the canvas
        canvas.paste(rotated_img, (x_offset, y_offset), rotated_img)

        # Draw the bounding box
        draw = ImageDraw.Draw(canvas)
        bbox = (x_offset, y_offset, x_offset + img.width, y_offset + img.height)
        # Rotate the bounding box by the same angle
        # Calculate the corners of the bounding box
        corners = [
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3])
        ]
        # Rotate each corner around the center of the image
        rotated_corners = []
        center_x = x_offset + img.width / 2
        center_y = y_offset + img.height / 2
        angle_rad = -torch.tensor(angle * (torch.pi / 180))  # Convert angle to radians

        for corner in corners:
            x, y = corner
            # Translate to origin (center of the image)
            x -= center_x
            y -= center_y
            # Rotate
            x_rotated = x * torch.cos(angle_rad) - y * torch.sin(angle_rad)
            y_rotated = x * torch.sin(angle_rad) + y * torch.cos(angle_rad)
            # Translate back
            x_rotated += center_x
            y_rotated += center_y
            rotated_corners.append((x_rotated, y_rotated))

        # Translate the corners to the expanded image position
        rotated_corners = [(x + delta_x / 2, y + delta_y / 2) for x, y in rotated_corners]
        list_of_bboxes.append(rotated_corners)
        
        # # Draw the rotated bounding box
        # draw.polygon(rotated_corners, outline='red', fill=None)

        # Update x_offset for the next image
        y_offset += rotated_img.height + random.randint(-MAX_PADDING, MAX_PADDING)  # Add some space between images

    canvas = canvas.crop((0, 0, canvas_width, y_offset + MAX_PADDING))  # Crop the canvas to the used area

    return canvas, list_of_bboxes