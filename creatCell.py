import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation


def generate_poisson_seeds(num_cells, grid_size, center, radius, min_dist):
    seeds = []
    attempts = 0

    while len(seeds) < num_cells and attempts < num_cells * 2000:
        p = np.random.randint(0, grid_size, size=3)
        if np.linalg.norm(p - center) > radius:
            attempts += 1
            continue

        ok = True
        for s in seeds:
            if np.linalg.norm(p - s) < min_dist:
                ok = False
                break

        if ok:
            seeds.append(p)

        attempts += 1

    return np.array(seeds)


def generate_cells(
    grid_size=101,
    sphere_radius=40,
    num_cells=20,
    nucleus_ratio=0.2,
    nucleus_offset_ratio=0.2,
    seed=0
):
    np.random.seed(seed)

    cell = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    center = np.array([grid_size // 2] * 3)

    # 座標
    xs, ys, zs = np.indices((grid_size, grid_size, grid_size))
    coords = np.stack([xs, ys, zs], axis=-1)

    dist_from_center = np.linalg.norm(coords - center, axis=-1)

    # 外形ノイズ（球の歪み）
    noise = gaussian_filter(np.random.normal(size=dist_from_center.shape), sigma=6)
    noise = noise / noise.std() * (sphere_radius * 0.05)
    deformed_radius = sphere_radius + noise

    sphere_mask = dist_from_center <= deformed_radius

    # --- seed生成（均一化） ---
    min_dist = sphere_radius / (num_cells ** (1/3)) * 1.5
    seeds = generate_poisson_seeds(num_cells, grid_size, center, sphere_radius, min_dist)

    # --- 重み（Power diagram） ---
    weights = np.random.uniform(0.9, 1.1, size=num_cells)

    # --- 距離ノイズ ---
    noise_field = gaussian_filter(
        np.random.normal(size=(grid_size, grid_size, grid_size)),
        sigma=5
    )
    noise_field = noise_field / noise_field.std()

    # --- Voronoi ---
    flat_coords = coords.reshape(-1, 3)
    dists = np.linalg.norm(flat_coords[:, None, :] - seeds[None, :, :], axis=2)

    # 重み適用
    dists = dists / weights[None, :]

    # ノイズ適用
    noise_flat = noise_field.reshape(-1, 1)
    dists = dists * (1 + 0.2 * noise_flat)

    labels = np.argmin(dists, axis=1).reshape(grid_size, grid_size, grid_size)

    # --- 丸め処理 ---
    new_labels = -np.ones_like(labels)

    for i in range(num_cells):
        mask = (labels == i) & sphere_mask
        if np.sum(mask) == 0:
            continue

        dist = distance_transform_edt(mask)

        if np.max(dist) == 0:
            continue

        threshold = np.percentile(dist[dist > 0], 60)
        rounded = dist > threshold * 0.4

        new_labels[rounded] = i

    labels = new_labels

    # --- 細胞生成 ---
    for i in range(num_cells):
        mask = (labels == i)
        indices = np.argwhere(mask)

        if len(indices) == 0:
            continue

        centroid = indices.mean(axis=0)
        dists = np.linalg.norm(indices - centroid, axis=1)
        r_cell = dists.max()

        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)

        r_nucleus = r_cell * nucleus_ratio

        max_offset = r_cell - r_nucleus
        scale = (np.random.uniform(0.0, 1.0))**2
        offset = direction * max_offset * scale * nucleus_offset_ratio
        nucleus_center = centroid + offset

        for idx in indices:
            d = np.linalg.norm(idx - nucleus_center)

            if d <= r_nucleus:
                cell[tuple(idx)] = 2
            else:
                cell[tuple(idx)] = 1

    # --- 細胞膜 ---
    membrane = np.zeros_like(cell, dtype=bool)

    directions = [(dx, dy, dz)
                  for dx in [-1, 0, 1]
                  for dy in [-1, 0, 1]
                  for dz in [-1, 0, 1]
                  if not (dx == dy == dz == 0)]

    for dx, dy, dz in directions:
        shifted = np.roll(labels, shift=(dx, dy, dz), axis=(0, 1, 2))
        boundary = (labels != shifted) & (labels >= 0)
        membrane |= boundary

    # 膜を薄く
    membrane = binary_dilation(membrane, iterations=1)

    cell[membrane & (cell != 2)] = 3

    # --- 外周膜 ---
    outer_membrane = np.zeros_like(cell, dtype=bool)
    for dx, dy, dz in directions:
        shifted_mask = np.roll(sphere_mask, shift=(dx, dy, dz), axis=(0, 1, 2))
        boundary_outer = sphere_mask & (~shifted_mask)
        outer_membrane |= boundary_outer

    cell[outer_membrane] = 3

    # 外は0
    cell[~sphere_mask] = 0

    return cell


# 実行
cell = generate_cells(
    grid_size=101,
    sphere_radius=40,
    num_cells=12,
    nucleus_ratio=0.15,
    nucleus_offset_ratio=0.2,
    seed=1
)

print(cell.shape)
print("値:", np.unique(cell))

np.save("cell.npy", cell)

