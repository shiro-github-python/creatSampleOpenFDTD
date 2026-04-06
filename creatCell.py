import numpy as np

def generate_voronoi_cells(
    grid_size=100,
    sphere_radius=40,
    num_cells=50,
    nucleus_ratio=0.3,
    nucleus_offset_ratio=0.5,
    seed=0
):
    np.random.seed(seed)

    cell = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    center = np.array([grid_size // 2] * 3)

    # 球内マスク
    xs, ys, zs = np.indices((grid_size, grid_size, grid_size))
    coords = np.stack([xs, ys, zs], axis=-1)
    dist_from_center = np.linalg.norm(coords - center, axis=-1)
    sphere_mask = dist_from_center <= sphere_radius

    # 細胞中心
    seeds = []
    while len(seeds) < num_cells:
        p = np.random.randint(0, grid_size, size=3)
        if np.linalg.norm(p - center) <= sphere_radius:
            seeds.append(p)
    seeds = np.array(seeds)

    # ボロノイ分割
    flat_coords = coords.reshape(-1, 3)
    dists = np.linalg.norm(flat_coords[:, None, :] - seeds[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1).reshape(grid_size, grid_size, grid_size)

    # 各細胞ごと処理
    for i in range(num_cells):
        mask = (labels == i) & sphere_mask
        indices = np.argwhere(mask)

        if len(indices) == 0:
            continue

        centroid = indices.mean(axis=0)
        dists = np.linalg.norm(indices - centroid, axis=1)
        r_cell = dists.max()

        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)

#        offset = direction * r_cell * nucleus_offset_ratio
#        scale = np.random.uniform(0.0, nucleus_offset_ratio)
#        offset = direction * r_cell * scale
#        nucleus_center = centroid + offset

        r_nucleus = r_cell * nucleus_ratio

        max_offset = r_cell - r_nucleus
        scale = (np.random.uniform(0.0, 1.0))**2  # 中心寄り分布

        offset = direction * max_offset * scale
        nucleus_center = centroid + offset

####################333
        
#direction = np.random.normal(size=3)
#direction /= np.linalg.norm(direction)

#r_nucleus = r_cell * nucleus_ratio

#max_offset = r_cell - r_nucleus
#scale = (np.random.uniform(0.0, 1.0))**2  # 中心寄り分布

#offset = direction * max_offset * scale
#nucleus_center = centroid + offset



########################33


        
        for idx in indices:
            d = np.linalg.norm(idx - nucleus_center)

            if d <= r_nucleus:
                cell[tuple(idx)] = 2  # 核
            else:
                cell[tuple(idx)] = 1  # 細胞質

    # -------------------------
    # 細胞膜の追加（ここが新規）
    # -------------------------
    membrane = np.zeros_like(cell, dtype=bool)

    # 6近傍方向
    directions = [
        (1,0,0), (-1,0,0),
        (0,1,0), (0,-1,0),
        (0,0,1), (0,0,-1),
        (1,1,0), (1,-1,0),
        (-1/np.sqrt(2),1/np.sqrt(2),0), (-1/np.sqrt(2),-1/np.sqrt(2),0),
        (1/np.sqrt(2),0,1/np.sqrt(2)), (1/np.sqrt(2),0,-1/np.sqrt(2)),
        (-1/np.sqrt(2),0,1/np.sqrt(2)), (-1/np.sqrt(2),0,-1/np.sqrt(2)),
        (0,1/np.sqrt(2),1/np.sqrt(2)), (0,1/np.sqrt(2),-1/np.sqrt(2)),          
        (0,-1/np.sqrt(2),1/np.sqrt(2)), (0,-1/np.sqrt(2),-1/np.sqrt(2)), 
        (1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)), (1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)),
        (1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)), (-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)),
        (1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)), (-1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)),
        (-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)), (-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3))
    ]

    for dx, dy, dz in directions:
        shifted = np.roll(labels, shift=(dx, dy, dz), axis=(0,1,2))
        boundary = (labels != shifted) & sphere_mask
        membrane |= boundary

    # 膜を上書き（核や細胞質より優先）
    cell[membrane] = 3

    # 球外は0
    cell[~sphere_mask] = 0

    # -------------------------
    # 外周（細胞外との境界）にも膜を追加
    # -------------------------
    outer_membrane = np.zeros_like(cell, dtype=bool)

    for dx, dy, dz in directions:
        shifted_mask = np.roll(sphere_mask, shift=(dx, dy, dz), axis=(0,1,2))
        boundary_outer = sphere_mask & (~shifted_mask)
        outer_membrane |= boundary_outer

    # 膜に追加（核は上書きしない場合）
    #cell[(outer_membrane) & (cell != 2)] = 3
    cell[(outer_membrane)] = 3
    
    return cell


# 実行例
cell = generate_voronoi_cells(
    grid_size=80,
    sphere_radius=30,
    num_cells=40,
    nucleus_ratio=0.2,
    nucleus_offset_ratio=0.3
)

print(cell.shape)
print("値:", np.unique(cell))
np.save("cell.npy", cell)
