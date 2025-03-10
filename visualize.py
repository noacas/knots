import numpy as np
from matplotlib import pyplot as plt, patches as patches
from matplotlib.path import Path

from components import get_n_strands


def trace_strand_path(word, strand_idx, n_strands, closure=True):
    """
    Trace the path of a single strand, including the closure path if requested.
    """
    current_pos = strand_idx
    path_points = [(0, current_pos)]

    # Trace through the braid
    for t, crossing in enumerate(word):
        i = abs(crossing) - 1
        if current_pos in [i, i + 1]:
            next_pos = i + 1 if current_pos == i else i
            x0, x1 = t, t + 1
            y0, y1 = current_pos, next_pos

            path_points.extend([
                (x0, y0),
                (x0 + 0.25, y0),
                (x0 + 0.5, (y0 + y1) / 2),
                (x0 + 0.75, y1),
                (x1, y1)
            ])
            current_pos = next_pos
        else:
            path_points.append((t + 1, current_pos))

    if closure:
        # Add closure path
        final_x = len(word)
        final_y = current_pos

        # Calculate control points for closure curve
        dx = final_x * 0.4  # Width of closure curve
        dy = n_strands * 0.4  # Height of closure curve

        # Add points for the closure curve
        path_points.extend([
            (final_x, final_y),
            (final_x + dx, final_y),
            (final_x + dx, strand_idx),
            (final_x, strand_idx)
        ])

    return path_points


def visualize_knotified_braid(word):
    """
    Visualize a braid with closure to show it as a knot.
    """
    n_strands = get_n_strands(word)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Adjust plot limits to accommodate closure
    ax.set_xlim(-0.5, len(word) + len(word) * 0.4)
    ax.set_ylim(-0.5, n_strands - 0.5)
    ax.axis('off')

    # Generate paths for each strand
    colors = plt.cm.rainbow(np.linspace(0, 1, n_strands))

    for strand, color in zip(range(n_strands), colors):
        points = trace_strand_path(word, strand, n_strands)

        # Convert points to path vertices and codes
        verts = []
        codes = []

        # Start the path
        verts.append(points[0])
        codes.append(Path.MOVETO)

        # Add the rest of the points
        for i in range(1, len(points)):
            verts.append(points[i])
            codes.append(Path.CURVE4 if i % 5 in [2, 3, 4] else Path.LINETO)

        path = Path(verts, codes)

        # Draw the strand with varying alpha for crossings
        for t, crossing in enumerate(word):
            i = abs(crossing) - 1
            is_over = crossing > 0

            # Find the y-coordinate at this x-position
            x = t + 0.5
            y_coords = [p[1] for p in points if abs(p[0] - x) < 0.1]
            if not y_coords:
                continue
            y = y_coords[0]

            if y in [i, i + 1]:
                alpha = 1.0 if (is_over and y == i) or (not is_over and y == i + 1) else 0.3
                patch = patches.PathPatch(path, facecolor='none', edgecolor=color, alpha=alpha, linewidth=2)
                ax.add_patch(patch)
                break
        else:
            patch = patches.PathPatch(path, facecolor='none', edgecolor=color, linewidth=2)
            ax.add_patch(patch)

    plt.title(f"Knotified Braid: {word}")
    return fig
