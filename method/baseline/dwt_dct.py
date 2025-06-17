import numpy as np
import pywt
from scipy.fftpack import dct, idct

def embed_watermark(tensor, watermark):
    """
    将水印嵌入到形状为 (4, 64, 64) 的张量中。
    
    参数：
    tensor: numpy 数组，形状为 (4, 64, 64)，表示 4 个灰度图像。
    watermark: numpy 数组，形状为 (32, 32)，表示要嵌入的水印图像。
    
    返回：
    watermarked_tensor: numpy 数组，形状为 (4, 64, 64)，表示嵌入水印后的张量。
    """
    watermarked_tensor = np.copy(tensor)
    wm_shape = watermark.shape

    for i in range(tensor.shape[0]):
        # 对每个图像进行 DWT 分解
        coeffs = pywt.dwt2(tensor[i], 'haar')
        LL, (LH, HL, HH) = coeffs

        # 对 LL 子带进行 DCT 变换
        dct_LL = dct(LL)

        # 将水印嵌入到 DCT 系数中
        dct_LL[:wm_shape[0], :wm_shape[1]] += watermark

        # 对修改后的 DCT 系数进行逆 DCT 变换
        LL_modified = idct(dct_LL)

        # 重构图像
        coeffs_modified = LL_modified, (LH, HL, HH)
        watermarked_image = pywt.idwt2(coeffs_modified, 'haar')

        # 将结果存储到 watermarked_tensor 中
        watermarked_tensor[i] = watermarked_image

    return watermarked_tensor

def extract_watermark(watermarked_tensor, watermark_shape):
    """
    从形状为 (4, 64, 64) 的张量中提取水印。
    
    参数：
    watermarked_tensor: numpy 数组，形状为 (4, 64, 64)，表示嵌入水印后的 4 个灰度图像。
    watermark_shape: tuple，表示水印的形状，例如 (32, 32)。
    
    返回：
    extracted_watermark: numpy 数组，形状为 (32, 32)，表示提取的水印图像。
    """
    extracted_watermark = np.zeros(watermark_shape)

    for i in range(watermarked_tensor.shape[0]):
        # 对每个水印图像进行 DWT 分解
        coeffs = pywt.dwt2(watermarked_tensor[i], 'haar')
        LL, (LH, HL, HH) = coeffs

        # 对 LL 子带进行 DCT 变换
        dct_LL = dct(LL)

        # 从 DCT 系数中提取水印信息
        extracted_watermark += dct_LL[:watermark_shape[0], :watermark_shape[1]]

    # 取平均值
    extracted_watermark /= watermarked_tensor.shape[0]

    return extracted_watermark

def similar(x, y):
    len1 = min(len(x), len(y))
    x = np.double(x[:len1])
    y = np.double(y[:len1])
    z = np.sum(x * y) / (np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2)))
    return z

# # 示例用法
# if __name__ == "__main__":
#     # 创建一个示例张量，形状为 (4, 64, 64)
#     tensor = np.random.rand(4, 64, 64) * 255

#     # 创建一个示例水印，形状为 (32, 32)
#     watermark = np.random.rand(32, 32) * 10  # 水印强度可根据需要调整

#     # 嵌入水印
#     watermarked_tensor = embed_watermark(tensor, watermark)

#     # 显示原始图像和嵌入水印后的图像
#     import matplotlib.pyplot as plt

#     for i in range(tensor.shape[0]):
#         plt.subplot(2, 4, i + 1)
#         plt.imshow(tensor[i], cmap='gray')
#         plt.title(f'Original Image {i+1}')
#         plt.axis('off')

#         plt.subplot(2, 4, i + 5)
#         plt.imshow(watermarked_tensor[i], cmap='gray')
#         plt.title(f'Watermarked Image {i+1}')
#         plt.axis('off')

#     plt.show()
