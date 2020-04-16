import numpy as np
import cv2
import os

import cfg
####TODO
def gen(batch_size=4, is_val=False):
    img_h, img_w = cfg.max_train_img_size, cfg.max_train_img_size
    # img_h, img_w =352,640

    pixel_num_h = img_h // cfg.pixel_size
    pixel_num_w = img_w // cfg.pixel_size

    y = np.zeros((batch_size,pixel_num_h, pixel_num_w,7), dtype=np.float32)

    if is_val:
        with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
            f_list = f_val.readlines()
    else:

        with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
            f_list = f_train.readlines()
    images = []
    while True:
        for i in xrange(batch_size):
            # random gen an image name
            random_img = np.random.choice(f_list)
            img_filename = str(random_img).strip().split(',')[0]

            img_path = os.path.join(cfg.data_dir,
                                    cfg.train_image_dir_name,
                                    img_filename)

            img = cv2.imread(img_path)
            if img == None:
                print(img_path)
                continue
            img = img.astype(np.float32)
            b, g, r = cv2.split(img)
            b -= 103.94
            g -= 116.78
            r -= 123.68
            img = cv2.merge((b, g, r))
            images.append(img[:, :, ::-1])


            gt_file = os.path.join(cfg.data_dir,
                                   cfg.train_label_dir_name,
                                   img_filename[:-4] + '_gt.npy')
            y[i] = np.load(gt_file)

            if len(images) == batch_size:
                return images,y


