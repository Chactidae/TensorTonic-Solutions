import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    if isinstance(feature_size, int):
        H_feat = W_feat = feature_size
    else:
        H_feat, W_feat = feature_size

    # 2. Приводим image_size к (H_img, W_img)
    if isinstance(image_size, int):
        H_img = W_img = image_size
    else:
        H_img, W_img = image_size

    # 3. Шаг сетки (stride) по высоте и ширине
    stride_y = H_img / H_feat
    stride_x = W_img / W_feat

    # 4. Центры ячеек в координатах изображения
    cy = (np.arange(H_feat) + 0.5) * stride_y   # для каждой строки
    cx = (np.arange(W_feat) + 0.5) * stride_x   # для каждого столбца
    cx_grid, cy_grid = np.meshgrid(cx, cy)      # shape (H_feat, W_feat)
    cells = np.stack([cx_grid, cy_grid], axis=-1).reshape(-1, 2)  # (H_feat*W_feat, 2)

    # 5. Все комбинации ширин и высот для (scale, ratio)
    scales = np.asarray(scales)
    ratios = np.asarray(aspect_ratios)
    sqrt_ratios = np.sqrt(ratios)

    # w и h – двумерные массивы (кол-во масштабов × кол-во отношений)
    w = np.outer(scales, sqrt_ratios)        # (S, R)
    h = np.outer(scales, 1.0 / sqrt_ratios)  # (S, R)
    w_flat = w.ravel()  # (S*R,)
    h_flat = h.ravel()  # (S*R,)

    n_cells = cells.shape[0]
    n_boxes_per_cell = len(scales) * len(ratios)

    # 6. Привязываем каждый anchor к своей ячейке
    centers = np.repeat(cells, n_boxes_per_cell, axis=0)  # (N, 2)
    cx_all = centers[:, 0:1]
    cy_all = centers[:, 1:2]

    w_all = np.tile(w_flat, n_cells).reshape(-1, 1)
    h_all = np.tile(h_flat, n_cells).reshape(-1, 1)

    # 7. Координаты боксов
    x1 = cx_all - w_all / 2
    y1 = cy_all - h_all / 2
    x2 = cx_all + w_all / 2
    y2 = cy_all + h_all / 2

    coords = [x1, y1, x2, y2]
    anchors = []
    print(coords)
    if len(x1) > 1:
        for i in range(len(x1)):
            anchor_i = []
            for j in range(len(coords)):
              
                anchor_i.append(coords[j][i][0])
    
            anchors.append(anchor_i)
    else:
        for i in range(len(coords)):
            
            anchors.append(coords[i][0][0])
        
        anchors = [anchors]
    #print(anchors)
    return anchors