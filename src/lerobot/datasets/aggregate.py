import logging
import shutil
import copy
import json
import numpy as np
from pathlib import Path
import pandas as pd
import tqdm
import multiprocessing
from functools import partial

# 仅保留基础日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Lerobot原生工具（仅保留必要的，无Metadata依赖）
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_PATH,
    to_parquet_with_hf_images,
)
from lerobot.datasets.video_utils import get_video_duration_in_s

# -------------------------- 【完全保留的3.0基础工具函数（无修改）】 --------------------------
def read_info_json(root: Path) -> dict:
    info_path = root / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"源数据集缺少info.json: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_info_json(info: dict, root: Path):
    info_path = root / "info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False, default=_numpy_to_list)

def read_stats_json(root: Path) -> dict:
    stats_path = root / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"源数据集缺少stats.json: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
        stats = _recursive_list_to_numpy(stats)
    return stats

def write_stats_json(stats: dict, root: Path):
    stats_path = root / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False, default=_numpy_to_list)

def read_tasks_parquet(root: Path) -> pd.DataFrame:
    tasks_path = root / "tasks.parquet"
    if not tasks_path.exists():
        logger.warning(f"源数据集缺少tasks.parquet，返回空DataFrame: {tasks_path}")
        return pd.DataFrame()
    return pd.read_parquet(tasks_path)

def write_tasks_parquet(tasks: pd.DataFrame, root: Path):
    if tasks.empty:
        logger.warning("任务列表为空，跳过tasks.parquet写入")
        return
    tasks_path = root / "tasks.parquet"
    tasks.to_parquet(tasks_path, index=True)

def _numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.int64, np.float_, np.float64, np.bool_)):
        return obj.item()
    raise TypeError(f"无法序列化类型: {type(obj)}")

def _recursive_list_to_numpy(obj):
    if isinstance(obj, list):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {k: _recursive_list_to_numpy(v) for k, v in obj.items()}
    else:
        return obj

def validate_all_datasets(src_roots: list[Path], base_info: dict):
    """
    基于基准info校验所有数据集，无重复读取第一个源的meta
    基准info来自第一个源，仅读一次
    """
    # 从基准info取校验基准
    fps = base_info["fps"]
    robot_type = base_info["robot_type"]
    features = base_info["features"]

    # 遍历所有源校验（从第二个开始，第一个已读）
    for src_root in tqdm.tqdm(src_roots, desc="校验所有源数据集属性"):
        src_meta_root = src_root / "meta"
        cur_info = read_info_json(src_meta_root)
        if cur_info["fps"] != fps:
            raise ValueError(f"数据集{src_root.name}fps不匹配：期望{fps}，实际{cur_info['fps']}")
        if cur_info["robot_type"] != robot_type:
            raise ValueError(f"数据集{src_root.name}robot_type不匹配：期望{robot_type}，实际{cur_info['robot_type']}")
        if cur_info["features"] != features:
            raise ValueError(f"数据集{src_root.name}features不匹配，字段不一致")
    logger.info(f"所有数据集属性校验通过 | fps={fps} | robot_type={robot_type}")

def read_src_episodes(src_root: Path) -> pd.DataFrame:
    """裸路径读取源episodes，修复路径拼接重复问题（移除多余的/meta）"""
    # 关键修复：DEFAULT_EPISODES_PATH 本身已包含 meta/episodes/，直接用src_root拼接
    ep_path = src_root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    if not ep_path.exists():
        raise FileNotFoundError(f"源数据集{src_root.name}缺少episodes文件: {ep_path}")
    return pd.read_parquet(ep_path)

def update_data_df(df: pd.DataFrame, src_tasks: pd.DataFrame, dst_tasks: pd.DataFrame, global_ep_offset: int, global_fr_offset: int):
    """数据索引全局偏移+Task映射，无Metadata依赖"""
    df["episode_index"] = df["episode_index"] + global_ep_offset
    df["index"] = df["index"] + global_fr_offset

    index_columns_to_map = [
        "task_index", "coarse_task_index", "coarse_quality_index", "quality_index", "operating_hand_index"
    ]
    for col in index_columns_to_map:
        if col not in df.columns or src_tasks.empty or dst_tasks.empty:
            continue
        old_indices = df[col].to_numpy()
        src_names = src_tasks.index.take(old_indices)
        new_indices = dst_tasks.loc[src_names, "task_index"].to_numpy()
        df[col] = new_indices
    return df

def update_meta_data(
    df: pd.DataFrame,
    global_ep_offset: int,
    global_fr_offset: int,
    meta_idx: dict,
    data_idx: dict,
    videos_idx: dict,
):
    """元数据索引全局偏移，1Dataset=1Chunk核心逻辑"""
    # Chunk索引偏移（仅加数据集对应的chunk索引，file保留源值）
    df["meta/episodes/chunk_index"] = df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]
    df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
    df["data/file_index"] = df["data/file_index"] + data_idx["file"]

    # 视频索引更新：chunk=数据集序号，file保留源值
    for key, video_idx in videos_idx.items():
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        if orig_chunk_col not in df.columns or orig_file_col not in df.columns:
            continue
        df["_orig_chunk"] = df[orig_chunk_col].copy()
        df["_orig_file"] = df[orig_file_col].copy()

        df[orig_chunk_col] = video_idx["chunk"]
        df[orig_file_col] = df["_orig_file"]

        # Timestamp偏移处理
        src_to_offset = video_idx.get("src_to_offset", {})
        if src_to_offset:
            for idx in df.index:
                src_key = (df.at[idx, "_orig_chunk"], df.at[idx, "_orig_file"])
                offset = src_to_offset.get(src_key, 0)
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
        else:
            df[f"videos/{key}/from_timestamp"] = 0
            df[f"videos/{key}/to_timestamp"] = df[f"videos/{key}/to_timestamp"] - df[f"videos/{key}/from_timestamp"]

        df = df.drop(columns=["_orig_chunk", "_orig_file"])

    # 全局EP/FR偏移
    df["dataset_from_index"] = df["dataset_from_index"] + global_fr_offset
    df["dataset_to_index"] = df["dataset_to_index"] + global_fr_offset
    df["episode_index"] = df["episode_index"] + global_ep_offset
    return df

def aggregate_videos(src_root: Path, dst_root: Path, videos_idx: dict):
    """视频仅复制不拼接，1Dataset=1Chunk，file保留源值"""
    src_episodes_df = read_src_episodes(src_root)
    for key, video_idx in videos_idx.items():
        chunk_col = f"videos/{key}/chunk_index"
        file_col = f"videos/{key}/file_index"
        if chunk_col not in src_episodes_df.columns or file_col not in src_episodes_df.columns:
            continue
        unique_pairs = sorted({(c, f) for c, f in zip(src_episodes_df[chunk_col], src_episodes_df[file_col], strict=False)})

        for src_c, src_f in unique_pairs:
            src_path = src_root / DEFAULT_VIDEO_PATH.format(video_key=key, chunk_index=src_c, file_index=src_f)
            if not src_path.exists():
                logger.warning(f"源视频不存在，跳过：{src_path}")
                continue
            # 目标路径：chunk=数据集序号，file=源file索引
            dst_path = dst_root / DEFAULT_VIDEO_PATH.format(video_key=key, chunk_index=video_idx["chunk"], file_index=src_f)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if not dst_path.exists():
                shutil.copy2(str(src_path), str(dst_path))
            video_idx["src_to_offset"][(src_c, src_f)] = 0
            video_idx["episode_duration"] = get_video_duration_in_s(src_path)
    return videos_idx

def aggregate_data(src_root: Path, dst_root: Path, data_idx: dict, src_tasks: pd.DataFrame, dst_tasks: pd.DataFrame, global_ep_offset: int, global_fr_offset: int):
    """数据Parquet处理，1Dataset=1Chunk，仅更新索引不拼接"""
    contains_images = False
    src_episodes_df = read_src_episodes(src_root)
    unique_pairs = sorted({(c, f) for c, f in zip(src_episodes_df["data/chunk_index"], src_episodes_df["data/file_index"], strict=False)})

    for src_c, src_f in tqdm.tqdm(unique_pairs, desc=f"处理数据 - {src_root.name}"):
        src_path = src_root / DEFAULT_DATA_PATH.format(chunk_index=src_c, file_index=src_f)
        if not src_path.exists():
            logger.warning(f"源数据不存在，跳过：{src_path}")
            continue
        df = pd.read_parquet(src_path)
        df = update_data_df(df, src_tasks, dst_tasks, global_ep_offset, global_fr_offset)
        # 目标路径：chunk=数据集序号，file=源file索引
        dst_path = dst_root / DEFAULT_DATA_PATH.format(chunk_index=data_idx["chunk"], file_index=src_f)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path)
        else:
            df.to_parquet(dst_path, index=False)
    if unique_pairs:
        data_idx["file"] = max([f for _, f in unique_pairs]) + 1
    return data_idx

def aggregate_metadata(src_root: Path, dst_root: Path, meta_idx: dict, data_idx: dict, videos_idx: dict, global_ep_offset: int, global_fr_offset: int):
    """元数据Parquet处理，1Dataset=1Chunk，仅更新索引不拼接"""
    src_episodes_df = read_src_episodes(src_root)
    unique_pairs = sorted({(c, f) for c, f in zip(src_episodes_df["meta/episodes/chunk_index"], src_episodes_df["meta/episodes/file_index"], strict=False)})

    for src_c, src_f in tqdm.tqdm(unique_pairs, desc=f"处理元数据 - {src_root.name}"):
        src_path = src_root / DEFAULT_EPISODES_PATH.format(chunk_index=src_c, file_index=src_f)
        if not src_path.exists():
            logger.warning(f"源元数据不存在，跳过：{src_path}")
            continue
        df = pd.read_parquet(src_path)
        df = update_meta_data(df, global_ep_offset, global_fr_offset, meta_idx, data_idx, videos_idx)
        # 目标路径：chunk=数据集序号，file=源file索引
        dst_path = dst_root / DEFAULT_EPISODES_PATH.format(chunk_index=meta_idx["chunk"], file_index=src_f)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_path, index=False)
    if unique_pairs:
        meta_idx["file"] = max([f for _, f in unique_pairs]) + 1
    return meta_idx

def process_single(src_root: Path, global_offset: tuple[int, int], dataset_idx: int, dst_root: Path, dst_tasks: pd.DataFrame, video_keys: list[str]):
    """单数据集处理核心：1Dataset=1Chunk，无Metadata依赖"""
    global_ep_offset, global_fr_offset = global_offset
    src_meta_root = src_root / "meta"
    src_tasks = read_tasks_parquet(src_meta_root)

    # 初始化索引：chunk_index=数据集序号（1Dataset=1Chunk）
    videos_idx = {key: {"chunk": dataset_idx, "src_to_offset": {}, "episode_duration": 0} for key in video_keys}
    data_idx = {"chunk": dataset_idx, "file": 0}
    meta_idx = {"chunk": dataset_idx, "file": 0}

    logger.info(f"开始处理 {src_root.name} | Chunk索引={dataset_idx} | EP偏移={global_ep_offset} | FR偏移={global_fr_offset}")
    aggregate_videos(src_root, dst_root, videos_idx)
    aggregate_data(src_root, dst_root, data_idx, src_tasks, dst_tasks, global_ep_offset, global_fr_offset)
    aggregate_metadata(src_root, dst_root, meta_idx, data_idx, videos_idx, global_ep_offset, global_fr_offset)
    logger.info(f"处理完成 {src_root.name} | Chunk索引={dataset_idx}")

    # 仅返回统计（主函数已提前收集，用于日志校验）
    src_info = read_info_json(src_meta_root)
    return {"name": src_root.name, "ep": src_info["total_episodes"], "fr": src_info["total_frames"], "chunk": dataset_idx}

def finalize_aggregation(
    dst_root: Path,
    base_info: dict,       # 第一个源的info模板（仅读一次）
    dst_tasks: pd.DataFrame,
    total_chunks: int,     # 总chunk数=数据集数量
    total_episodes: int,   # 提前累加的全局总EP
    total_frames: int,     # 提前累加的全局总FR
    all_src_stats: list[dict],  # 提前收集的源stats
    chunk_size: int        # Chunk划分粒度（Lerobot规范）
):
    """
    最终收尾：完全贴合2.1逻辑，从第一个源取info模板，仅修改指定key
    无任何重复读取，所有数据均为提前收集的已知数据
    """
    dst_meta_root = dst_root / "meta"
    dst_meta_root.mkdir(parents=True, exist_ok=True)

    # 1. 写入全局Tasks
    print("Start writing global tasks.parquet...")
    write_tasks_parquet(dst_tasks, dst_meta_root)

    # 2. 核心：复用第一个源的info模板，仅更新指定关键字段（2.1逻辑）
    final_info = copy.deepcopy(base_info)  # 深拷贝模板，不修改原数据
    final_info.update({
        "total_tasks": len(dst_tasks),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_chunks": total_chunks,      # 总chunk数=数据集数量
        "chunks_size": chunk_size,         # 正确的Chunk划分粒度，非1
        "splits": {"train": f"0:{total_episodes}"},  # 全局split
    })
    write_info_json(final_info, dst_meta_root)

    # 3. 聚合并写入全局stats
    if all_src_stats:
        global_stats = aggregate_stats(all_src_stats)
        write_stats_json(global_stats, dst_meta_root)

    # 打印最终统计
    logger.info(
        f"全局元数据写入完成 | 总Chunk={total_chunks} | 总EP={total_episodes} "
        f"| 总FR={total_frames} | 总Task={len(dst_tasks)} | Chunk粒度={chunk_size}"
    )

# -------------------------- 【主函数：整合所有逻辑，仅读一次meta，无冗余】 --------------------------
def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    chunk_size: int | None = None,
):
    logger.info("===== 开始数据集合并【1Dataset=1Chunk+无重复IO】 =====")
    # 初始化参数
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    aggr_root = Path(aggr_root).absolute() if aggr_root else Path.cwd() / aggr_repo_id
    aggr_root.mkdir(parents=True, exist_ok=True)
    src_roots = [Path(root).absolute() for root in roots] if roots else []
    if len(repo_ids) != len(src_roots):
        raise ValueError(f"repo_ids({len(repo_ids)})与roots({len(src_roots)})数量不匹配")
    if len(src_roots) == 0:
        raise ValueError("无待合并的源数据集")
    total_chunks = len(src_roots)  # 总chunk数=数据集数量，1Dataset=1Chunk

    first_src_root = src_roots[0]
    first_meta_root = first_src_root / "meta"
    base_info = read_info_json(first_meta_root)  # 仅读一次，全程复用
    logger.info(f"成功读取第一个源[{first_src_root.name}]的Info作为模板，开始校验")


    validate_all_datasets(src_roots, base_info)

    video_keys = [k for k in base_info["features"] if base_info["features"][k]["dtype"] == "video"]
    logger.info(f"提取视频Keys：{video_keys if video_keys else '无视频数据'}")

    all_src_tasks = []
    for src_root in src_roots:
        src_tasks = read_tasks_parquet(src_root / "meta")
        if not src_tasks.empty:
            all_src_tasks.append(src_tasks)
    unique_tasks = pd.concat(all_src_tasks).index.unique() if all_src_tasks else pd.Index([])
    dst_tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)
    logger.info(f"生成全局Tasks | 总数={len(dst_tasks)} | 任务列表={unique_tasks.tolist()}")

    global_offsets = []
    total_episodes = 0
    total_frames = 0
    all_src_stats = []
    for src_root in src_roots:
        src_meta_root = src_root / "meta"
        # 仅读一次：当前源的info和stats
        src_info = read_info_json(src_meta_root)
        src_stats = read_stats_json(src_meta_root)
        # 记录全局偏移
        global_offsets.append((total_episodes, total_frames))
        # 累加全局统计
        total_episodes += src_info["total_episodes"]
        total_frames += src_info["total_frames"]
        # 收集stats
        all_src_stats.append(src_stats)
    logger.info(f"全局偏移预计算完成 | 总EP={total_episodes} | 总FR={total_frames} | 已收集{len(all_src_stats)}个源stats")

    process_args = []
    for idx, (src_root, offset) in enumerate(zip(src_roots, global_offsets)):
        process_args.append((
            src_root, offset, idx, aggr_root, dst_tasks, video_keys
        ))

    multiprocessing.set_start_method("fork", force=True)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"启动多进程 | 进程数={num_processes} | 待处理数据集={total_chunks} | 总Chunk数={total_chunks}")
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 收集处理结果（用于日志校验）
        results = list(tqdm.tqdm(
            pool.starmap(partial(process_single), process_args),
            total=len(process_args),
            desc="并行处理所有数据集"
        ))

    # 打印多进程处理结果汇总
    logger.info("===== 多进程处理结果汇总 =====")
    for res in results:
        logger.info(f"数据集{res['name']} | Chunk{res['chunk']} | 处理EP={res['ep']} | 处理FR={res['fr']}")

    finalize_aggregation(
        dst_root=aggr_root,
        base_info=base_info,
        dst_tasks=dst_tasks,
        total_chunks=total_chunks,
        total_episodes=total_episodes,
        total_frames=total_frames,
        all_src_stats=all_src_stats,
        chunk_size=chunk_size
    )

    logger.info("===== 数据集合并全部完成 =====")
    logger.info(f"合并后路径：{aggr_root.absolute()}")
    logger.info(f"核心规范：1Dataset=1Chunk | 总Chunk数={total_chunks} | Info模板复用第一个源")
    return aggr_root, results
