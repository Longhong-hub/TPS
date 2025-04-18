from keras.layers import BatchNormalizationV2, add
from keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply, Activation
kernel_initializer = 'he_uniform'

def se_block(x, reduction_ratio=16):
    """
    SE-Net 注意力模块
    :param x: 输入特征图 (batch, h, w, c)
    :param reduction_ratio: 降维比例
    :return: 注意力增强后的特征图
    """
    # 全局平均池化
    avg_pool = GlobalAveragePooling2D()(x)
    # 第一全连接层（降维）
    fc1 = Dense(int(x.shape[-1] // reduction_ratio), activation='relu')(avg_pool)
    # 第二全连接层（升维）
    fc2 = Dense(x.shape[-1], activation='sigmoid')(fc1)
    # 注意力加权
    return Multiply()([x, fc2])

def conv_block_2D(x, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
    result = x

    for i in range(0, repeat):

        if block_type == 'thirdscope':
            result = thirdscope_conv2D_block(result, filters, size=size, padding=padding)
        elif block_type == 'Dual_Path':
            result = Dual_Path_conv2D_block(result, filters, size=size)
        elif block_type == 'firstscope':
            result = firstscope_conv2D_block(result, filters)
        elif block_type == 'secondscope':
            result = secondscope_conv2D_block(result, filters)
        elif block_type == 'EResnet':
            result = se_block(result, reduction_ratio=16)
            result = resnet_conv2D_block(result, filters, dilation_rate)
        elif block_type == 'conv':
            result = Conv2D(filters, (size, size),
                            activation='relu', kernel_initializer=kernel_initializer, padding=padding)(result)
        elif block_type == 'double_convolution':
            result = double_convolution_with_batch_normalization(result, filters, dilation_rate)

        else:
            return None

    return result

def MWS_conv2D_block(x, filters, size):
   
    x1 = firstscope_conv2D_block(x, filters)

    x2 = secondscope_conv2D_block(x, filters)

    x3 = thirdscope_conv2D_block(x, filters, size=6, padding='same')

    x = add([x1, x2, x3])


    return x

def FAM_conv2D_block(x, filters, size):

    x4 = conv_block_2D(x, filters, 'EResnet', repeat=1)

    x5 = conv_block_2D(x, filters, 'EResnet', repeat=2)

    x6 = conv_block_2D(x, filters, 'EResnet', repeat=3)


    x = add([x4, x5, x6])


    return x


def Dual_Path_conv2D_block(x, filters, size):
    x = BatchNormalizationV2(axis=-1)(x)
    mmor_output = MWS_conv2D_block(x, filters, size)
    fam_output = FAM_conv2D_block(x, filters, size)
    
    # 将两个模块的输出相加
    x = add([mmor_output, fam_output])


    x = BatchNormalizationV2(axis=-1)(x)

    return x


def thirdscope_conv2D_block(x, filters, size=3, padding='same'):
    x = Conv2D(filters, (1, size), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (size, 1), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    return x


def firstscope_conv2D_block(x, filters):
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=1)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=2)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    return x


def secondscope_conv2D_block(x, filters):
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=1)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=2)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=3)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    return x


def resnet_conv2D_block(x, filters, dilation_rate=1):
    x1 = Conv2D(filters, (1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same',
                dilation_rate=dilation_rate)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)
    x_final = add([x, x1])

    x_final = BatchNormalizationV2(axis=-1)(x_final)

    return x_final


def double_convolution_with_batch_normalization(x, filters, dilation_rate=1):
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    return x
