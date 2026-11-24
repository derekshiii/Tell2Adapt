import os
import numpy as np
from PIL import Image
from pathlib import Path

def convert_npz_to_png(input_dir, output_dir):
    """
    将NPZ格式的医学影像预测转换为PNG文件

    参数:
        input_dir: NPZ文件所在目录
        output_dir: PNG输出目录
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有NPZ文件
    input_path = Path(input_dir)
    npz_files = list(input_path.glob('*.npz'))

    if len(npz_files) == 0:
        print(f"在 {input_dir} 中未找到NPZ文件")
        return

    print(f"找到 {len(npz_files)} 个NPZ文件")

    # 处理每个NPZ文件
    for npz_file in npz_files:
        try:
            # 加载NPZ文件
            data = np.load(npz_file)

            # 检查是否存在'segs'键
            if 'segs' not in data:
                print(f"警告: {npz_file.name} 中未找到'segs'键,跳过")
                continue

            # 获取分割预测
            segs = data['segs']

            print(f"处理: {npz_file.name}, shape: {segs.shape}")

            # 如果是3通道,取第一个通道或使用argmax
            if len(segs.shape) == 3 and segs.shape[2] == 3:
                # 假设是one-hot编码或多通道预测,使用argmax获取类别
                segs = np.argmax(segs, axis=2)
            elif len(segs.shape) == 3:
                # 如果是其他3D格式,取第一个通道
                segs = segs[:, :, 0]

            # 创建映射后的图像
            output_img = np.zeros_like(segs, dtype=np.uint8)

            # 映射: 1 -> 127, 2 -> 254
            output_img[segs == 1] = 127
            output_img[segs == 2] = 254
            # 0保持为0(背景)

            # 转换为PIL图像并保存
            img = Image.fromarray(output_img, mode='L')

            # 生成输出文件名(保持原名,改扩展名为.png)
            output_filename = npz_file.stem + '.png'
            output_path = Path(output_dir) / output_filename

            img.save(output_path)

            # 统计信息
            unique_values = np.unique(segs)
            print(f"  已保存: {output_filename}")
            print(f"  原始类别: {unique_values}")
            print(f"  映射后像素值: {np.unique(output_img)}")

        except Exception as e:
            print(f"处理 {npz_file.name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n转换完成! 输出目录: {output_dir}")

# 使用示例
if __name__ == "__main__":
    input_directory = "/path/to/Inference_results/RIGA"  # NPZ文件所在目录
    output_directory = "/path/to/Inference_results/RIGA_png"  # PNG输出目录

    convert_npz_to_png(input_directory, output_directory)
