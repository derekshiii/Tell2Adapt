import os
import numpy as np
from PIL import Image
import argparse


def overlay_mask_on_image(image_path, mask_path, output_path, alpha=0.3, color=(204, 51, 51)):
    """
    将预测标签以指定透明度和颜色叠加到医学影像上

    参数:
        image_path: 原始医学影像路径
        mask_path: 预测标签路径
        output_path: 输出路径
        alpha: 透明度 (0-1之间，默认0.3)
        color: RGB颜色元组 (默认为(204, 51, 51))
    """
    # 读取原始图像
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)

    # 读取预测标签
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    # 创建彩色叠加层
    overlay = np.zeros_like(img_array)

    # 找到非零区域（预测的标签区域）
    mask_binary = mask_array > 0

    # 将指定颜色应用到标签区域
    overlay[mask_binary] = color

    # 将叠加层与原图混合
    result = img_array.copy().astype(float)
    result[mask_binary] = (1 - alpha) * img_array[mask_binary] + alpha * overlay[mask_binary]
    result = result.astype(np.uint8)

    # 保存结果
    result_image = Image.fromarray(result)
    result_image.save(output_path)
    print(f"已保存: {output_path}")


def batch_overlay(image_dir, mask_dir, output_dir, alpha=0.3, color=(204, 51, 51)):
    """
    批量处理图像和标签的叠加

    参数:
        image_dir: 原始医学影像目录
        mask_dir: 预测标签目录
        output_dir: 输出目录
        alpha: 透明度 (0-1之间，默认0.3)
        color: RGB颜色元组 (默认为(204, 51, 51))
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有PNG文件
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    print(f"找到 {len(image_files)} 个图像文件")

    # 处理每个图像
    for img_file in image_files:
        image_path = os.path.join(image_dir, img_file)

        # 移除_0000后缀来匹配标签文件
        # 例如: image_0001_0000.png -> image_0001.png
        mask_file = img_file.replace('_0000.png', '.png')
        mask_path = os.path.join(mask_dir, mask_file)
        output_path = os.path.join(output_dir, img_file)

        # 检查对应的标签文件是否存在
        if not os.path.exists(mask_path):
            print(f"警告: 未找到对应的标签文件 {mask_file}（原图像: {img_file}），跳过")
            continue

        try:
            overlay_mask_on_image(image_path, mask_path, output_path, alpha, color)
        except Exception as e:
            print(f"处理 {img_file} 时出错: {str(e)}")

    print(f"\n处理完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将医学影像预测标签叠加到原图上')
    parser.add_argument('--image_dir', type=str, default='/path/to/nnUNet_raw/Dataset182_CVCDB/imagesTr', help='原始医学影像目录')
    parser.add_argument('--mask_dir', type=str, default='/path/to/DFG/BiomedParse_test/CVCDB_Inference', help='预测标签目录')
    parser.add_argument('--output_dir', type=str, default='/path/to/visualizations/CVCDB_Overlay', help='输出目录')
    parser.add_argument('--alpha', type=float, default=0.5, help='透明度 (0-1之间，默认0.3)')
    parser.add_argument('--color', type=int, nargs=3, default=[ 51,204,204],
                        help='RGB颜色值 (默认: 204 51 51)')

    args = parser.parse_args()

    batch_overlay(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
        color=tuple(args.color)
    )
