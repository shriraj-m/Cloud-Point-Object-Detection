import matplotlib.pyplot as plt
import numpy as np

def bev_projection(pointcloud, res=0.1, x_range=(-40,40), y_range=(-40,40)):
    """ 
    Convert point cloud to Bird's Eye View for easier visualization
    Instead of dealing with 3D point clouds, we can use a 2D BEV map to make
    the object detection easier. We can map 3D points onto a 2D map using matplotlib
    and using their height (so the z value) as a pixel intensity

    This code should filter the points within a specific range (40), map them to
    a 2D grid indices, and fill in pixel values using the z as the height.
    """

    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]

    # Filter points within given x and y range.
    mask = (x > x_range[0]) & (x < x_range[1]) & (y > y_range[0]) & (y < y_range[1])
    x = x[mask]
    y = y[mask]
    z = z[mask]

    # Now convert into image coordinates
    x_image = np.floor((x - x_range[0]) / res).astype(int)
    y_image = np.floor((y - y_range[0]) / res).astype(int)

    # Now create Bird's Eye View Image
    temp_x = (int(x_range[1] - x_range[0]) / res)
    temp_y = (int(y_range[1] - y_range[0]) / res)

    bev_map = np.zeros(temp_x, temp_y)
    bev_map[x_image, y_image]

    plt.imshow(bev_map, cmap='gray')
    plt.show()