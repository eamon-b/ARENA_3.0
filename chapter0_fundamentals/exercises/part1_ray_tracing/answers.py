# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
from plotly_utils import imshow

MAIN = __name__ == "__main__"

# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    # so all origins are the same, x is the same, want linspace in y...
    rays = torch.zeros(num_pixels, 2, 3)
    # extract the x coord - all pixels, 2nd point, first (x) coord
    rays[:, 1, 0] = 1
    # extract y coord
    y = rays[:, 1, 1]
    # edit view in place
    torch.linspace(-y_limit, y_limit, num_pixels, out=y)
    return rays

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
# %%
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=list(range(10)))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)
# %%
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    # ignore the z coord
    O = ray[0, :-1]
    D = ray[1, :-1]
    L1_minus_L2 = segment[0, :-1] - segment[1, :-1]
    matrix = torch.stack((D, L1_minus_L2), dim=1) # NOTE: dim=1!
    assert matrix.shape == (2, 2)
    b = segment[0, :-1] - O
    assert b.shape == (2,)
    try:
        solution = torch.linalg.solve(matrix, b)
        assert solution.shape == (2,)
        return solution[0] >= 0 and 0 <= solution[1] <= 1
    except RuntimeError as e:
        # matrix not invertible, no solutions
        print(f"Solve failed: {e}")
        return False


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    # TODO: add type hints for each tensor? for clarity
    nrays = rays.shape[0]
    nsegments = segments.shape[0]

    # ignore the z coord and make consistent batches - first 2 dims nrays, nsegments
    rays_repeated: Float[Tensor, "nrays nsegments 2 2"] = einops.repeat(rays[..., :2], "a b c -> a repeat b c", repeat=nsegments)
    segments_repeated: Float[Tensor, "nrarys nsegments 2 2"] = einops.repeat(segments[..., :2], "a b c -> repeat a b c", repeat=nrays)
    assert segments_repeated.shape == (nrays, nsegments, 2, 2)

    # extract O, D, L1, L2
    Os: Float[Tensor, "nrarys nsegments 2"] = rays_repeated[:, :, 0]
    Ds: Float[Tensor, "nrarys nsegments 2"] = rays_repeated[:, :, 1]

    L1s: Float[Tensor, "nrarys nsegments 2"] = segments_repeated[:, :, 0]
    L2s: Float[Tensor, "nrarys nsegments 2"] = segments_repeated[:, :, 1]
    assert L1s.shape == (nrays, nsegments, 2)

    # make matrix and B to solve
    matrices: Float[Tensor, "nrays nsegments 2 2"] = torch.stack((Ds, (L1s - L2s)), dim=-1)
    assert matrices.shape == (expected := (nrays, nsegments, 2, 2)), f"matrices is wrong shape: got {matrices.shape}, expected {expected}"
    # where the matrix is _not_ invertible, replace with identity
    invertible = matrices.det().abs() > 1e-8
    is_singular = ~invertible
    matrices[is_singular] = torch.eye(2) # replace _not_ invertible
    bs: Float[Tensor, "nrays nsegments 2"] = L1s - Os
    assert bs.shape == (nrays, nsegments, 2)

    # solve
    solutions = torch.linalg.solve(matrices, bs)
    assert solutions.shape == (nrays, nsegments, 2)

    # check which solutions meet our criteria
    u = solutions[..., 0]
    v = solutions[..., 1]
    satisfy_solutions = (0 <= u) & (0 <= v) & (v <= 1)

    # make sure it was a real solve to begin with
    satisfy_solutions &= invertible
    satisfy_solutions = satisfy_solutions.any(dim=1)
    assert satisfy_solutions.shape == (expected := (nrays,)), f"satisfy_solutions is wrong shape: got {satisfy_solutions.shape}, expected {expected}"
    return satisfy_solutions




tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    nrays = num_pixels_y * num_pixels_z
    rays = torch.zeros(nrays, 2, 3)
    rays[:, 1, 0] = 1
    ys = torch.linspace(-y_limit, y_limit, num_pixels_y)
    rays[:, 1, 1] = einops.repeat(ys, "a -> (a repeat)", repeat=num_pixels_z) # repeat each thing
    zs = torch.linspace(-z_limit, z_limit, num_pixels_z)
    rays[:, 1, 2] = einops.repeat(ys, "a -> (repeat a)", repeat=num_pixels_y) # repeat whole thing

    return rays


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
# %%
one_triangle = t.tensor([[0, 0, 0], [4, 0.5, 0], [2, 3, 0]])
A, B, C = one_triangle
x, y, z = one_triangle.T

fig: go.FigureWidget = setup_widget_fig_triangle(x, y, z)
display(fig)


@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def update(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.update_traces({"x": [P[0]], "y": [P[1]]}, 2)
# %%
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    matrix = torch.stack((-D, B - A, (C - A)), dim=-1)
    assert matrix.shape == (3, 3)
    b = O - A
    if matrix.det().abs() < 1e-8: # singular
        return False
    solution = torch.linalg.solve(matrix, b)
    s, u, v = solution
    return 0 <= s and 0 <= u and 0 <= v and u + v <= 1


tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    nrays = rays.shape[0]
    Os, Ds = rays.unbind(1)
    # nice way of getting triangles easily:
    A, B, C = einops.repeat(triangle, "pts dims -> pts repeat dims", repeat=nrays)
    matrix = torch.stack((-Ds, B - A, C - A), dim=-1)
    b = Os - A
    assert matrix.shape == (nrays, 3, 3)
    assert b.shape == (nrays, 3)

    is_singular = matrix.det().abs() < 1e-8
    matrix[is_singular] = torch.eye(3)
    solutions = torch.linalg.solve(matrix, b)
    assert solutions.shape == (nrays, 3)
    s, u, v = solutions.unbind(dim=1)
    satisfy_solutions = (0 <= s) & (0 <= u) & (0 <= v) & (u + v <= 1) & ~is_singular

    return satisfy_solutions


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
# practice debugging
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(1) # was -1.. didn't get any useful error tho

    mat = t.stack([- D, B - A, C - A], dim=-1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return (0 <= s) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)
# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]
    rays = einops.repeat(rays, "a b c -> a repeat b c", repeat=ntriangles)
    triangles = einops.repeat(triangles, "a b c -> repeat a b c", repeat=nrays)
    O, D = rays.unbind(2)
    A, B, C = triangles.unbind(2)
    matrix = torch.stack((-D, B - A, C - A), dim=-1)
    assert matrix.shape == (nrays, ntriangles, 3, 3)
    is_singular = matrix.det().abs() < 1e-8
    matrix[is_singular] = torch.eye(3)
    b = O - A
    assert b.shape == (nrays, ntriangles, 3)
    solutions = torch.linalg.solve(matrix, b)
    assert solutions.shape == (nrays, ntriangles, 3)
    s, u, v = solutions.unbind(-1)
    satisfy = (0 <= s) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~satisfy] = float('inf')
    # Get the minimum distance (over all triangles) for each ray
    return einops.reduce(s, "nrays ntriangles -> nrays", "min")
    # return s.amin(dim=1)



num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()
# %%
def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    return torch.Tensor([
        [t.cos(theta), 0, t.sin(theta)],
        [0, 1, 0],
        [-t.sin(theta), 0, t.cos(theta)]
    ])


tests.test_rotation_matrix(rotation_matrix)
# %%
def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z] element is distance
    to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video").show()


num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)

display_video(dists)
# %%
def raytrace_mesh_gpu(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    All computations should be performed on the GPU.
    """
    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]
    device = "cuda"
    rays = rays.to(device)
    triangles = triangles.to(device)
    rays = einops.repeat(rays, "a b c -> a repeat b c", repeat=ntriangles)
    triangles = einops.repeat(triangles, "a b c -> repeat a b c", repeat=nrays)
    O, D = rays.unbind(2)
    A, B, C = triangles.unbind(2)
    matrix = torch.stack((-D, B - A, C - A), dim=-1)
    assert matrix.shape == (nrays, ntriangles, 3, 3)
    is_singular = matrix.det().abs() < 1e-8
    matrix[is_singular] = torch.eye(3)
    b = O - A
    assert b.shape == (nrays, ntriangles, 3)
    solutions = torch.linalg.solve(matrix, b)
    assert solutions.shape == (nrays, ntriangles, 3)
    s, u, v = solutions.unbind(-1)
    satisfy = (0 <= s) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~satisfy] = float('inf')
    # Get the minimum distance (over all triangles) for each ray
    result = einops.reduce(s, "nrays ntriangles -> nrays", "min")
    return result.cpu()


dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh_gpu, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)
display_video(dists)
# %%
def raytrace_mesh_lambert(
    rays: Float[Tensor, "nrays points=2 dims=3"],
    triangles: Float[Tensor, "ntriangles points=3 dims=3"],
    light: Float[Tensor, "dims=3"],
    ambient_intensity: float,
    device: str = "cuda",
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the intensity of light hitting the triangle it intersects with (or zero if no intersection).

    Args:
        rays:               A tensor of rays, with shape `[nrays, 2, 3]`.
        triangles:          A tensor of triangles, with shape `[ntriangles, 3, 3]`.
        light:              A tensor representing the light vector, with shape `[3]`. We compue the intensity as the dot
                            product of the triangle normals & the light vector, then set it to be zero if the sign is
                            negative.
        ambient_intensity:  A float representing the ambient intensity. This is the minimum brightness for a triangle,
                            to differentiate it from the black background (rays that don't hit any triangle).
        device:             The device to perform the computation on.

    Returns:
        A tensor of intensities for each of the rays, flattened over the [y, z] dimensions. The values are zero when
        there is no intersection, and `ambient_intensity + intensity` when there is an interesection (where `intensity`
        is the dot product of the triangle's normal vector and the light vector, truncated at zero).
    """
    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]
    rays = rays.to(device)
    triangles = triangles.to(device)

    rays_repeated = einops.repeat(rays, "nrays pts dims -> pts nrays repeat dims", repeat=ntriangles)
    triangles_repeated = einops.repeat(triangles, "ntriangles pts dims -> pts repeat ntriangles dims", repeat=nrays)
    O, D = rays_repeated
    A, B, C = triangles_repeated
    matrix = torch.stack((-D, B - A, C - A), dim=-1)
    assert matrix.shape == (nrays, ntriangles, 3, 3)
    is_singular = matrix.det().abs() < 1e-8
    matrix[is_singular] = torch.eye(3)
    b = O - A
    assert b.shape == (nrays, ntriangles, 3)
    solutions = torch.linalg.solve(matrix, b)
    assert solutions.shape == (nrays, ntriangles, 3)
    s, u, v = solutions.unbind(-1)
    satisfy = (0 <= s) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~satisfy] = float('inf')
    # Get the minimum distance (over all triangles) for each ray
    closest_distances, closest_triangles_for_each_ray = s.min(dim=-1)  # both shape [NR]

    # now we do something different - dot product of normal of triangle with light vector - just use triangles here, not repteated
    normals = torch.cross(triangles[:, 2] - triangles[:, 0], triangles[:, 1] - triangles[:, 0], dim=1) # shape [ntriangles dims]
    # normalise normals and light
    normals /= normals.norm(dim=1, keepdim=True)
    light /= light.norm()
    intensity_per_triangle = normals @ light.to(device)
    intensity_per_triangle_signed = t.where(intensity_per_triangle > 0, intensity_per_triangle, 0.0)

    # was missing this step of getting intensity just from closest triangle
    intensity = intensity_per_triangle_signed[closest_triangles_for_each_ray] + ambient_intensity

    # Set to zero if the ray doesn't intersect with any triangle
    intensity = t.where(closest_distances.isfinite(), intensity, 0.0)

    return intensity.cpu()


def display_video_with_lighting(intensity: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z] element is the
    lighting intensity based on the angle of light & the surface of the triangle which this ray hits first.
    """
    px.imshow(
        intensity,
        animation_frame=0,
        origin="lower",
        color_continuous_scale="magma",
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video (lighting)").show()


ambient_intensity = 0.5
light = t.tensor([0.0, -1.0, 1.0])
raytrace_function = partial(raytrace_mesh_lambert, ambient_intensity=ambient_intensity, light=light)

intensity = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_function, num_frames)
intensity = einops.rearrange(intensity, "frames (y z) -> frames y z", y=num_pixels_y)
display_video_with_lighting(intensity)
# %%
# some tests thanks to Gemini 2.5 pro, with critique from Claude
import pytest
import torch.nn.functional as F
# --- Helper Functions ---

def compute_triangle_normal(triangle: torch.Tensor) -> torch.Tensor:
    """Compute normal vector for a triangle using cross product."""
    # Triangle shape: [3, 3] - three vertices with xyz coordinates
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]
    # Cross product of two edges gives normal
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = torch.cross(edge1, edge2)
    return F.normalize(normal, dim=0)

# --- Test Fixtures ---

@pytest.fixture
def basic_triangle():
    """Provides a simple triangle on the XY plane."""
    # Triangle on XY plane with vertices ordered counter-clockwise from above
    # This gives normal vector pointing in +Z direction: [0, 0, 1]
    # Vertices: A(-1,-1,0), B(1,-1,0), C(0,1,0)
    return torch.tensor([
        [-1.0, -1.0, 0.0],  # Vertex A
        [ 1.0, -1.0, 0.0],  # Vertex B
        [ 0.0,  1.0, 0.0],  # Vertex C
    ])

@pytest.fixture
def triangle_at_z():
    """Factory for creating triangles at different Z coordinates."""
    def _triangle_at_z(z_coord=0.0, flip_normal=False):
        vertices = torch.tensor([
            [-1.0, -1.0, z_coord],
            [ 1.0, -1.0, z_coord], 
            [ 0.0,  1.0, z_coord],
        ])
        if flip_normal:
            # Reverse vertex order to flip normal
            vertices = vertices[[0, 2, 1]]
        return vertices.unsqueeze(0)  # Add batch dimension
    return _triangle_at_z

@pytest.fixture
def normalized_light():
    """Provides normalized light directions."""
    def _light(direction):
        return F.normalize(torch.tensor(direction, dtype=torch.float32), dim=0)
    return _light

@pytest.fixture
def scene_data(basic_triangle, normalized_light):
    """Provides common scene data for tests."""
    return {
        "triangles": basic_triangle.unsqueeze(0),  # Add batch dimension
        "light": normalized_light([0.5, 0.0, 1.0]),  # Light from above and side
        "ambient_intensity": 0.5,
        "device": "cpu",
    }

@pytest.fixture
def device():
    """Provides available device (CPU or CUDA if available)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

# --- Input Validation Tests ---

def test_invalid_ray_shape():
    """Test that function raises appropriate errors for invalid ray shapes."""
    triangles = torch.randn(1, 3, 3)
    light = F.normalize(torch.randn(3), dim=0)
    
    with pytest.raises((ValueError, RuntimeError)):
        # Wrong ray shape - should be [nrays, 2, 3]
        invalid_rays = torch.randn(2, 3)  # Missing origin/direction dimension
        raytrace_mesh_lambert(invalid_rays, triangles, light, 0.5)

def test_invalid_triangle_shape():
    """Test that function raises appropriate errors for invalid triangle shapes."""
    rays = torch.randn(1, 2, 3)
    light = F.normalize(torch.randn(3), dim=0)
    
    with pytest.raises((ValueError, RuntimeError)):
        # Wrong triangle shape - should be [ntriangles, 3, 3]
        invalid_triangles = torch.randn(1, 2, 3)  # Only 2 vertices
        raytrace_mesh_lambert(rays, invalid_triangles, light, 0.5)

def test_empty_inputs():
    """Test behavior with empty inputs."""
    # Empty rays
    empty_rays = torch.empty(0, 2, 3)
    triangles = torch.randn(1, 3, 3)
    light = F.normalize(torch.randn(3), dim=0)
    
    with pytest.raises((ValueError, RuntimeError)) if not pytest.mark.xfail else pytest.mark.xfail(raises=NotImplementedError):
        result = raytrace_mesh_lambert(empty_rays, triangles, light, 0.5)
        assert result.shape == (0,)

# --- Core Functionality Tests ---
def test_ray_misses_all_triangles(scene_data):
    """Test that a ray missing all triangles returns intensity 0.0."""
    # Ray parallel to XY-plane at z=1, pointing in +X direction
    # Triangle is at z=0, so this ray will never intersect
    ray = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])  # [origin, direction]

    scene_data_copy = scene_data.copy()
    result = raytrace_mesh_lambert(rays=ray, **scene_data_copy)
    
    expected = torch.tensor([0.0])
    assert torch.allclose(result, expected)
    assert result.shape == (1,)


def test_ray_hits_triangle_directly(scene_data):
    """Test direct intersection with proper illumination calculation."""
    # Ray starts at z=1 and points straight down (-Z direction)
    ray = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])  # [origin, direction]

    scene_data_copy = scene_data.copy()
    
    # Manual calculation of expected intensity:
    # Triangle normal computed from vertices: cross((B-A), (C-A))
    triangle = scene_data_copy["triangles"][0]  # Remove batch dimension
    triangle_normal = compute_triangle_normal(triangle)
    
    # Verify our assumption about the normal direction
    expected_normal = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(triangle_normal, expected_normal), f"Normal is {triangle_normal}, expected {expected_normal}"
    
    # Diffuse intensity = max(0, dot(normal, light))
    diffuse_intensity = torch.clamp(torch.dot(triangle_normal, scene_data_copy["light"]), min=0.0)
    expected_intensity = scene_data_copy["ambient_intensity"] + diffuse_intensity
    expected = torch.tensor([expected_intensity])

    result = raytrace_mesh_lambert(rays=ray, **scene_data_copy)
    assert torch.allclose(result, expected, atol=1e-6)


def test_ray_hits_back_face_of_triangle(scene_data, normalized_light):
    """Test that back-face illumination is properly handled (dot product < 0)."""
    # Ray hits triangle directly from above
    ray = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    
    scene_data_copy = scene_data.copy()
    # Light shining from below the triangle (negative Z direction)
    scene_data_copy["light"] = normalized_light([0.0, 0.0, -1.0])

    # Triangle normal is [0,0,1], light is [0,0,-1]
    # Dot product = -1, which should be clamped to 0
    expected_intensity = scene_data_copy["ambient_intensity"] + 0.0
    expected = torch.tensor([expected_intensity])
    
    result = raytrace_mesh_lambert(rays=ray, **scene_data_copy)
    assert torch.allclose(result, expected, atol=1e-6)


def test_multiple_rays_some_hit_some_miss(scene_data):
    """Test batching with mixed hit/miss results."""
    rays = torch.tensor([
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],   # Ray 1: Hits triangle (down from above)
        [[5.0, 5.0, 1.0], [1.0, 0.0, 0.0]],    # Ray 2: Misses (starts far away, points in +X)
        [[-10.0, 0.0, 1.0], [1.0, 0.0, 0.0]],  # Ray 3: Misses (starts far left, points in +X)
    ])
    
    scene_data_copy = scene_data.copy()
    
    # Expected for Ray 1 (Hit)
    triangle = scene_data_copy["triangles"][0]
    triangle_normal = compute_triangle_normal(triangle)
    diffuse_intensity = torch.clamp(torch.dot(triangle_normal, scene_data_copy["light"]), min=0.0)
    intensity_hit = scene_data_copy["ambient_intensity"] + diffuse_intensity
    
    # Expected for Rays 2,3 (Miss)
    intensity_miss = 0.0
    
    expected = torch.tensor([intensity_hit, intensity_miss, intensity_miss])
    
    result = raytrace_mesh_lambert(rays=rays, **scene_data_copy)
    
    assert result.shape == (3,)
    assert torch.allclose(result, expected, atol=1e-6)


def test_ray_intersects_closest_triangle(scene_data, triangle_at_z):
    """Test that ray hits the closest triangle when multiple intersections possible."""
    # Create a second triangle further away with opposite normal
    far_triangle = triangle_at_z(z_coord=-2.0, flip_normal=True)
    
    # Combine triangles
    all_triangles = torch.cat([scene_data["triangles"], far_triangle], dim=0)

    # Ray pointing down the Z-axis, passing through both triangles
    ray = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    
    scene_data_copy = scene_data.copy()
    scene_data_copy["triangles"] = all_triangles
    
    # Should hit the FIRST triangle (at z=0)
    close_triangle_normal = compute_triangle_normal(scene_data["triangles"][0])
    diffuse_intensity = torch.clamp(torch.dot(close_triangle_normal, scene_data_copy["light"]), min=0.0)
    expected_intensity = scene_data_copy["ambient_intensity"] + diffuse_intensity
    expected = torch.tensor([expected_intensity])
    
    result = raytrace_mesh_lambert(rays=ray, **scene_data_copy)
    assert torch.allclose(result, expected, atol=1e-6)

# --- Parametric Tests ---
@pytest.mark.parametrize("ambient_intensity", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ambient_intensity_variations(scene_data, ambient_intensity):
    """Test that ambient intensity is properly applied across different values."""
    ray = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])  # Ray that hits
    
    scene_data_copy = scene_data.copy()
    scene_data_copy["ambient_intensity"] = ambient_intensity
    
    result = raytrace_mesh_lambert(rays=ray, **scene_data_copy)
    
    # Result should be at least ambient_intensity for a hit
    assert result[0] >= ambient_intensity
    
    # Test miss case - should be exactly 0
    miss_ray = torch.tensor([[[5.0, 5.0, 1.0], [1.0, 0.0, 0.0]]])
    miss_result = raytrace_mesh_lambert(rays=miss_ray, **scene_data_copy)
    assert torch.allclose(miss_result, torch.tensor([0.0]))


@pytest.mark.parametrize("light_direction", [
    [0.0, 0.0, 1.0],   # From above (maximum intensity)
    [1.0, 0.0, 0.0],   # From side (should be 0 for this triangle)
    [0.0, 0.0, -1.0],  # From below (should be 0, back-face)
    [0.5, 0.5, 1.0],   # Diagonal
])
def test_light_direction_variations(scene_data, normalized_light, light_direction):
    """Test illumination with different light directions."""
    ray = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])  # Ray that hits
    
    scene_data_copy = scene_data.copy()
    scene_data_copy["light"] = normalized_light(light_direction)
    
    result = raytrace_mesh_lambert(rays=ray, **scene_data_copy)
    
    # Calculate expected intensity
    triangle = scene_data_copy["triangles"][0]
    triangle_normal = compute_triangle_normal(triangle)
    diffuse_intensity = torch.clamp(torch.dot(triangle_normal, scene_data_copy["light"]), min=0.0)
    expected_intensity = scene_data_copy["ambient_intensity"] + diffuse_intensity
    
    assert torch.allclose(result, torch.tensor([expected_intensity]), atol=1e-6)

# --- Device Tests ---
def test_cpu_device(scene_data):
    """Test computation on CPU device."""
    ray = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    
    scene_data_copy = scene_data.copy()
    scene_data_copy["device"] = "cpu"
    
    result = raytrace_mesh_lambert(rays=ray, **scene_data_copy)
    assert result.device.type == "cpu"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device(scene_data):
    """Test computation on CUDA device."""
    ray = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]]).cuda()
    
    scene_data_copy = scene_data.copy()
    # Move all tensors to CUDA
    scene_data_copy["triangles"] = scene_data_copy["triangles"].cuda()
    scene_data_copy["light"] = scene_data_copy["light"].cuda()
    scene_data_copy["device"] = "cuda"
    
    result = raytrace_mesh_lambert(rays=ray, **scene_data_copy)
    assert result.device.type == "cuda"

# --- Edge Case Tests ---
def test_ray_grazing_triangle_edge(basic_triangle):
    """Test ray that grazes triangle edge."""
    triangles = basic_triangle.unsqueeze(0)
    light = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    
    # Ray that grazes the edge from vertex A to vertex B
    ray = torch.tensor([[[0.0, -1.0, 1.0], [0.0, 0.0, -1.0]]])  # Hits edge at y=-1
    
    result = raytrace_mesh_lambert(ray, triangles, light, 0.5)
    # Should either hit or miss consistently (implementation dependent)
    assert result.shape == (1,)


def test_degenerate_triangle():
    """Test with degenerate triangle (collinear vertices)."""
    # Degenerate triangle - all vertices on a line
    degenerate = torch.tensor([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], 
        [2.0, 0.0, 0.0],
    ]]).unsqueeze(0)
    
    ray = torch.tensor([[[0.5, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    light = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    
    # Should handle gracefully (likely return 0 or raise appropriate error)
    with pytest.raises((ValueError, RuntimeError)) if not pytest.mark.xfail else pytest.mark.xfail(raises=NotImplementedError):
        result = raytrace_mesh_lambert(ray, degenerate, light, 0.5)

# --- Numerical Precision Tests ---
def test_numerical_precision():
    """Test that function handles floating point precision appropriately."""
    # Very small triangle
    tiny_triangle = torch.tensor([[
        [1e-6, 1e-6, 0.0],
        [2e-6, 1e-6, 0.0],
        [1.5e-6, 2e-6, 0.0],
    ]]).unsqueeze(0)
    
    ray = torch.tensor([[[1.5e-6, 1.5e-6, 1e-6], [0.0, 0.0, -1.0]]])
    light = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    
    result = raytrace_mesh_lambert(ray, tiny_triangle, light, 0.5)
    assert result.shape == (1,)
    # Should be either 0 (miss) or ambient + diffuse (hit)
    assert result[0] >= 0.0
# %%
if MAIN:
    pytest.main()
# %%
