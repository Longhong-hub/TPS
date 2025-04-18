import glob
import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm

folder_path = ""  # 填入你的数据目录路径

def load_data(img_height, img_width, images_to_be_loaded, dataset):
    IMAGES_PATH = folder_path + 'images/'
    MASKS_PATH = folder_path + 'masks/'

    # 修改点：将 kvasir 的扩展名从 .jpg 改为 .jpeg
    if dataset == 'BKAI-IGH NeoPolyp-tumor':
        train_ids = glob.glob(IMAGES_PATH + "*.jpeg")

    if dataset == 'bkai-igh-neopolyp':
        train_ids = glob.glob(IMAGES_PATH + "*.jpeg")

    if dataset == 'kvasir':
        train_ids = glob.glob(IMAGES_PATH + "*.jpg")  # 扩展名改为 jpeg

    if dataset == 'cvc-clinicdb':
        train_ids = glob.glob(IMAGES_PATH + "*.tif")

    if dataset == 'cvc-colondb' or dataset == 'etis-laribpolypdb':
        train_ids = glob.glob(IMAGES_PATH + "*.png")

    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)

    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

    print('Resizing training images and masks: ' + str(images_to_be_loaded))
    for n, id_ in tqdm(enumerate(train_ids)):
        if n == images_to_be_loaded:
            break

        image_path = id_
        mask_path = image_path.replace("images", "masks")

        image = imread(image_path)
        mask_ = imread(mask_path)

        mask = np.zeros((img_height, img_width), dtype=np.bool_)

        pillow_image = Image.fromarray(image)
        pillow_image = pillow_image.resize((img_height, img_width))  # 注意：这里应保持 (width, height) 顺序
        image = np.array(pillow_image)

        X_train[n] = image / 255

        pillow_mask = Image.fromarray(mask_)
        pillow_mask = pillow_mask.resize((img_height, img_width), resample=Image.LANCZOS)
        mask_ = np.array(pillow_mask)
        
        #原版
        #for i in range(img_height):
            #for j in range(img_width):
                #if mask_[i, j] >= 127:
                    #mask[i, j] = 1
                    
        #Y_train[n] = mask

       # 优化点：使用矢量化操作替代循环
        mask = (mask_ >= 127)

        ###将图片转化为灰度图
        mask = (mask_[:, :, 0] >= 127).astype(np.uint8)
        Y_train[n] = mask

    Y_train = np.expand_dims(Y_train, axis=-1)

    return X_train, Y_train