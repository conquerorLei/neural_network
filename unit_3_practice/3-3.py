from PIL import Image
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 打开图像
img = Image.open("lena.tiff")
# 分离成三个颜色通道
img_r, img_g, img_b = img.split()
# 设置画布及其大小
plt.figure(figsize=(10, 10))

# R通道处理
plt.subplot(221)
plt.axis("off")
img_small = img_r.resize((50, 50))
plt.imshow(img_small, cmap="gray")
plt.title("R-缩放", fontsize=14)

# G通道处理
plt.subplot(222)
plt.axis("off")
# 先水平翻转再逆时针旋转270度
img_tran = img_g.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
plt.imshow(img_tran, cmap="gray")
plt.title("G-镜像+旋转", fontsize=14)

# B通道处理
plt.subplot(223)
plt.axis("off")
# (0,0)-(150,150)无法得出给定图例图例
# img_region = img_b.crop((0,0,150,150))
# 右下角改为300，300则正常
img_region = img_b.crop((0, 0, 300, 300))
plt.imshow(img_region, cmap="gray")
plt.title("B-裁剪", fontsize=14)

# 合成
plt.subplot(224)
plt.axis("off")
img_rgb = Image.merge("RGB", [img_r, img_g, img_b])
img_rgb.save("test.png")
plt.imshow(img_rgb)
plt.title("RGB", fontsize=14)

# 显示
plt.show()
