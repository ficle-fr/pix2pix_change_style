import cairo
import numpy as np
import cv2
import math
import os

import argparse

LINE_LABEL = 1
ARC_LABEL = 2

w = 320
h = 320

max_lines = 10
max_ellipses = 10
max_rectangles = 10

def generate_lines_coords(n_lines, w, h):
    #xyxy
    xs = np.random.uniform(0, w, n_lines * 2).astype(int)
    ys = np.random.uniform(0, h, n_lines * 2).astype(int)
    xyxy = np.stack([xs, ys], axis = -1).reshape(-1, 4)
    return xyxy

def generate_arcs_coords(n_arcs, w, h):
    #xyxy
    min_r = w / 30
    max_r = w / 7
    rs = np.random.uniform(min_r, max_r, n_arcs)
    aas = np.random.uniform(math.pi / 2, math.pi, n_arcs)
    sas = np.random.uniform(0, math.pi, n_arcs)
    xs = np.random.uniform(max_r + 10, w - max_r - 10, n_arcs).astype(int)
    ys = np.random.uniform(0, h, n_arcs).astype(int)
    coords = np.stack([xs, ys, rs, sas, sas + aas], axis = -1)
    return coords

def get_boxes_from_lines(lines):
    #line - xyxy
    xs = lines[:, [0, 2]]
    ys = lines[:, [1, 3]]
    return np.stack([np.min(xs, axis = -1), np.min(ys, axis = -1), 
                     np.max(xs, axis = -1), np.max(ys, axis = -1)], axis = -1)

def get_box_from_mask(mask):
    #line - xyxy
    where = np.where(mask > 0)
    return np.array([np.min(where[1]), np.min(where[0]), 
                     np.max(where[1]), np.max(where[0])])

def generate_lines(coords, w, h, width, dash = [1.]):

    masks_line = np.zeros((len(coords), h, w))

    for mask_i, line in zip(range(len(masks_line)), coords):
        surface_line = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
        ctx_line = cairo.Context(surface_line)
        ctx_line.move_to(line[0], line[1])
        ctx_line.line_to(line[2], line[3])
        ctx_line.set_dash(dash)
        ctx_line.set_line_width(width)
        ctx_line.stroke()
        masks_line[mask_i][np.ndarray(shape=(h, w), dtype=np.uint8,
                                      buffer = surface_line.get_data()) > 0] = 1
    return masks_line.astype(np.uint8)

def generate_arcs(coords, w, h, width, dash = [1.]):
    masks_arc = np.zeros((len(coords), h, w))

    for mask_i, arc in zip(range(len(masks_arc)), coords):
        surface_arc = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
        ctx_arc = cairo.Context(surface_arc)
        ctx_arc.arc(arc[0], arc[1], arc[2], arc[3], arc[4])
        #ctx_arc.set_line_width(2)
        ctx_arc.set_dash(dash)
        ctx_arc.set_line_width(width)
        ctx_arc.stroke()
        masks_arc[mask_i][np.ndarray(shape=(h, w), dtype=np.uint8,
                                      buffer = surface_arc.get_data()) > 0] = 1
    return masks_arc.astype(np.uint8)


def generate_data():
    #for i in range(100):
    n_lines = np.random.uniform(1, 4, 1).astype(int)
    n_lines = 1
    lines_coords = generate_lines_coords(n_lines, w, h)
    #lines_boxes = get_boxes_from_lines(lines_coords).astype(np.float32)

    n_arcs = np.random.uniform(1, 4, 1).astype(int)
    n_arcs = 0
    arcs_coords = generate_arcs_coords(n_arcs, w, h)
    
    lines_masks = generate_lines(lines_coords, w, h, 5)
    arcs_masks = generate_arcs(arcs_coords, w, h, 5)
    
    masks = np.concatenate([lines_masks, arcs_masks], axis = 0)
    use1 = ~np.all(masks.reshape(masks.shape[0], -1) == 0, axis = -1)
    masks = masks[use1]
    
    bboxes = np.stack([get_box_from_mask(m) for m in masks], axis = 0)
    use2 = np.all(np.stack([bboxes[:, 0] != bboxes[:, 2], bboxes[:, 1] != bboxes[:, 3]], axis = -1), axis = -1)
    masks = masks[use2]
    bboxes = bboxes[use2]
    

    main_img = np.any(masks, axis = 0).astype(np.float32)
    main_img = 1 - main_img
    main_img = np.tile(main_img[np.newaxis], (3, 1, 1))

    labels = np.concatenate([np.full(n_lines, LINE_LABEL, dtype = np.int64),
                             np.full(n_arcs, ARC_LABEL, dtype = np.int64)], axis = 0)
    labels = labels[use1]
    labels = labels[use2]
    target = {'boxes': bboxes, 'masks': masks.astype(np.uint8), 
              'labels': labels}
    #yield (main_img, target)
    return (main_img, target)

def generate_data_yolo():
    #for i in range(100):
    n_lines = np.random.uniform(1, 5, 1).astype(int)
    #n_lines = 3
    lines_coords = generate_lines_coords(n_lines, w, h)
    #lines_boxes = get_boxes_from_lines(lines_coords).astype(np.float32)

    n_arcs = np.random.uniform(1, 5, 1).astype(int)
    #n_arcs = 2
    arcs_coords = generate_arcs_coords(n_arcs, w, h)
    
    lines_masks = generate_lines(lines_coords, w, h, 5)
    arcs_masks = generate_arcs(arcs_coords, w, h, 5)
    
    masks = np.concatenate([lines_masks, arcs_masks], axis = 0)
    labels = np.concatenate([np.full(n_lines, 0), np.full(n_arcs, 1)])

    def get_cnt(img):
        img = cv2.blur(img, (3,3))
        img = cv2.Canny(img, 100, 200)
        cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnts

    cnts = [get_cnt(mask) for mask in (1 - masks) * 255]
    use_mask = np.array([True if len(cnt) == 2 else False for cnt in cnts])
    labels = labels[use_mask]

    cnts = [cnt[1].reshape(-1, 2) for cnt in cnts if len(cnt) == 2]
    cnts = [cnt / [w, h] for cnt in cnts]
    cnts = [np.r_[cnt, cnt[0][np.newaxis]] for cnt in cnts]

        
    lines_imgs = generate_lines(lines_coords, w, h, 2)
    arcs_imgs = generate_arcs(arcs_coords, w, h, 2)
    imgs = np.concatenate([lines_imgs, arcs_imgs], axis = 0)
    imgs = imgs[use_mask]

    main_img = (1 - np.any(imgs, axis = 0)).astype(np.uint8) * 255
    return (main_img, labels, cnts)

def img_pair_gen1(w, h):
    """
    Generates a pair of images first of which is an image with randomly placed 
    arcs and lines and the second is the same image but is drawn using thicker 
    lines.

    Parameters:
    - w (int): width
    - h (int): height

    yield:
    pair: two images
    """

    while True:
        n_lines = np.random.uniform(1, 5, 1).astype(int)
        lines_coords = generate_lines_coords(n_lines, w, h)

        n_arcs = np.random.uniform(1, 5, 1).astype(int)
        arcs_coords = generate_arcs_coords(n_arcs, w, h)
            
        lines_imgs_1 = generate_lines(lines_coords, w, h, 2)
        arcs_imgs_1 = generate_arcs(arcs_coords, w, h, 2)
        imgs_1 = np.concatenate([lines_imgs_1, arcs_imgs_1], axis = 0)

        lines_imgs_2 = generate_lines(lines_coords, w, h, 1)
        arcs_imgs_2 = generate_arcs(arcs_coords, w, h, 5, [3., 3.])
        imgs_2 = np.concatenate([lines_imgs_2, arcs_imgs_2], axis = 0)


        in_img = (0.5 - np.any(imgs_1, axis = 0)).astype(np.float32) * 2

        out_img = (0.5 - np.any(imgs_2, axis = 0)).astype(np.float32) * 2

        yield (np.tile(in_img[..., np.newaxis], (1, 1, 3)),
               np.tile(out_img[..., np.newaxis], (1, 1, 3)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest = "image_folder", default = "images",
                        help = "Images folder")
    parser.add_argument("-l", dest = "label_folder", default = "labels", help = "Labels folder")
    args = parser.parse_args()

    for i in range(10000):
        img, labs, cnts = generate_data_yolo()
        fname = str(i)
        cv2.imwrite(os.path.join(args.image_folder, fname + ".png"), img)
        with open(os.path.join(args.label_folder, fname + ".txt"), 'w') as file:
            for lab, cnt in zip(labs, cnts):
                file.write("{l} {cx}\n".
                    format(l = lab, cx = " ".join(str(el) for el in cnt.flatten())))
