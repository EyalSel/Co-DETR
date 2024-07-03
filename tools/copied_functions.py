"""
This is functionality that's copy-pasted from the EfficientDet-Inference repo.
Doing this avoids doing a pip install, which might ruin the current co-detr
environment and slow down development. This should be addressed at some point.
"""
import pickle
import re
from pathlib import Path


def waymo_scenario_to_tuple(video_segment_str):
    """
    Parse a scenario name string schema and return parts as a tuple
    """
    rex = re.compile(r"(.+)-S(\d{1,2})")
    rex_full = re.compile(r"(.+)-S(\d{1,2})-P(\d{1,2})_(\d{1,2})")
    str_match = rex.fullmatch(video_segment_str)
    str_match_full = rex_full.fullmatch(video_segment_str)
    assert (str_match is not None or
            str_match_full is not None), video_segment_str
    x = video_segment_str.split("-S")
    chunk = x[0]
    chunk_number = int(chunk.split("_")[1])
    split = chunk.split("_")[0]
    rest = x[1]
    if str_match is not None:
        scenario = int(rest)
        segment = total_segments = None
    else:
        x = rest.split("-P")
        scenario = int(x[0])
        segment_part = x[1]
        segment_split = segment_part.split("_")
        segment = int(segment_split[0])
        total_segments = int(segment_split[1])
    return chunk_number, scenario, segment, total_segments, split


def scenario_to_path(scenario_str, dataset="waymo"):
    """
    Given a name of a scenario, return its path in as saved in
    ad-config-search's cloud bucket
    """
    if dataset == "waymo":
        (chunk_number, scenario, _, _,
         split) = waymo_scenario_to_tuple(scenario_str)
        if split == "validation":
            subdir_name = "validation"
        elif chunk_number <= 5:
            subdir_name = "training00_05"
        elif chunk_number <= 12:
            subdir_name = "training06_12"
        elif chunk_number <= 18:
            subdir_name = "training13_18"
        elif chunk_number <= 24:
            subdir_name = "training19_24"
        elif chunk_number <= 31:
            subdir_name = "training25_31"
        else:
            raise Exception(f"Invalid chunk number {scenario_str}")
        run_path = (f"waymo_pl/{subdir_name}/"
                    f"{split}_{str(chunk_number).zfill(4)}/S{scenario}.pl")
    elif dataset == "argoverse":
        sector, segment = scenario_str.split("-")
        assert sector == "val" or sector.startswith("train_"), sector
        if sector == "val":
            run_path = f"argo_qdtrack/argo_val_qdtrack_as_label/{segment}.pl"
        else:
            sector_number = sector.split("_")[1]
            run_path = "argo_qdtrack/argo_{}_qdtrack_as_label/{}.pl".format(
                sector_number, segment)
    elif dataset == "MEVA":
        with open("meva_dataset_lookup.pl", 'rb') as f:
            lookup = pickle.load(f)
        split, scenario = scenario_str.split("--")
        run_path = lookup[scenario + ".avi"] + "/" + scenario + ".avi"
        run_path = run_path[1:]  # get rid of leading /
        run_path = str(Path("MEVA") / run_path)
    elif dataset == "UAVDT":
        return f"UAV-benchmark-M/{scenario_str.split('-')[1]}"
    elif dataset == "DETRAC":
        split, scenario = scenario_str.split("-")
        run_path = ("Insight-MVT_Annotation_" + split.capitalize() + "/" +
                    scenario)
    elif dataset == "VIRAT":
        split, scenario = scenario_str.split("-")
        run_path = ("VIRAT/" + split + "/" + scenario + ".mp4")
    elif dataset == "bdd100k":
        splits = scenario_str.split("-")
        assert len(splits) == 3, splits
        split = splits[0]
        scenario = splits[1] + "-" + splits[2]
        run_path = "bdd100k/images/track/" + split + "/" + scenario
    elif dataset == "visdrone":
        split, scenario = scenario_str.split("-")
        run_path = (f"VisDrone2019-MOT-{split}/sequences/{scenario}")
    else:
        raise Exception(f"Unknown dataset name {dataset}")
    return run_path


class OfflineWaymoSensorV1_1():
    """
    Same as version 1 except _child_index_dir is run offline first
    """

    def __init__(self, data_path):
        with open(data_path, 'rb') as handle:
            dicts = pickle.load(handle)
        self.all_data = dicts

    def total_num_frames(self):
        return len(self.all_data)

    def get_frame(self, frame_index):
        return self.all_data[frame_index]


class FrameFraction:
    """
    Specifies the crop location in the frame, as well as the final height and
    scale that the crop should be resized to.
    """

    def __init__(self, ymin, xmin, ymax, xmax, height=None, width=None):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax
        self.height = height
        self.width = width


class FractioningSchema:
    """
    Abstract parent class.
    """

    def get_split_specs(self, h, w):
        """
        Takes frame height and width and returns a list of FrameFraction
        instances.
        """
        raise NotImplementedError("Implemented by child class")
