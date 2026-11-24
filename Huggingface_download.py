import os
from huggingface_hub import snapshot_download

def download_hf_dataset_subset(repo_id, folder_in_repo, local_dir, repo_type='dataset'):
    """
    从Hugging Face Hub上下载数据集的特定子目录。

    Args:
        repo_id (str): 数据集的仓库ID，例如 "junma/CVPR-BiomedSegFM"。
        folder_in_repo (str): 仓库中您想下载的文件夹路径。
        local_dir (str): 文件要保存到的本地目录路径。
        repo_type (str, optional): 仓库类型，对于数据集应为 'dataset'。
    """
    print(f"开始从仓库 '{repo_id}' 下载子目录 '{folder_in_repo}'...")
    print(f"文件将保存至: {os.path.abspath(local_dir)}")

    # 使用 allow_patterns 来指定只下载目标文件夹下的所有文件
    # 模式 "folder/*" 表示匹配 folder 目录下的任何文件
    allow_patterns = [f"{folder_in_repo}/*"]

    try:
        # 使用 snapshot_download 函数进行下载
        # 它会自动处理缓存、断点续传和显示进度条
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,  # 设置为False，直接将文件复制到目标目录，而不是使用符号链接
            resume_download=True
        )
        print("\n下载完成！")
        print(f"数据集子目录已成功保存到 '{os.path.abspath(local_dir)}'")

    except Exception as e:
        print(f"\n下载过程中发生错误: {e}")
        print("请检查以下几点：")
        print(f"1. 仓库ID '{repo_id}' 是否正确？")
        print(f"2. 文件夹路径 '{folder_in_repo}' 是否存在于仓库中？")
        print("3. 您的网络连接是否稳定？")
        print("4. 是否已通过 'huggingface-cli login' 登录？")

if __name__ == "__main__":
    # --- 请在这里配置您的参数 ---

    # 1. Hugging Face上的数据集仓库ID
    DATASET_REPO_ID = "junma/CVPR-BiomedSegFM"

    # 2. 您想下载的仓库内部的文件夹路径
    FOLDER_TO_DOWNLOAD = "3D_train_npz_all/CT/CT_TotalSeg_organs" #    3D_train_npz_all/MRI/MR_BraTS-T2f  3D_train_npz_all/MRI/MR_BraTS-T2w
    # 3D_train_npz_all/MRI/MR_TotalSeg  3D_train_npz_all/CT/CT_TotalSeg_organs 3D_train_npz_all/CT/CT_TotalSeg_muscles
    # 3. 您想将文件保存到的本地目录
    #    脚本会自动创建这个目录
    LOCAL_SAVE_PATH = "/path/to/Datasets/CT_TotalSeg_organs"

    # --- 配置结束 ---

    # 运行下载函数
    download_hf_dataset_subset(
        repo_id=DATASET_REPO_ID,
        folder_in_repo=FOLDER_TO_DOWNLOAD,
        local_dir=LOCAL_SAVE_PATH
    )
