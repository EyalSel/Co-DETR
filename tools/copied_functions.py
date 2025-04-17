"""
This is functionality that's copy-pasted from the EfficientDet-Inference repo.
Doing this avoids doing a pip install, which might ruin the current co-detr
environment and slow down development. This should be addressed at some point.
"""
import multiprocessing
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
from google.cloud import storage

cv2.setNumThreads(0)
multiprocessing.set_start_method('spawn', force=True)


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
        with open("dataset_scenarios/meva_dataset_lookup.pl", 'rb') as f:
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
    elif dataset == "kitti-step":
        splits = scenario_str.split("-")
        assert len(splits) == 2, splits
        split, scenario = splits
        assert split in ["training", "validation", "testing"], split
        middle_path = {
            "training": "training/image_02",
            "validation": "training/image_02",
            "testing": "testing/image_02",
        }[split]
        run_path = f"kitti-step_material/{middle_path}/{scenario}"
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


class MEVASensor:

    def __init__(self, data_path):
        cap = cv2.VideoCapture(str(data_path))
        self.cap = cap
        self.total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def total_num_frames(self):
        return self.total_length

    def get_frame(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        assert ret, frame_index
        return {"center_camera_feed": frame}

    def __del__(self):
        self.cap.release()


class ImageDirectorySensor:

    def __init__(self,
                 data_path,
                 clean_fn=lambda x: x.replace("img", ""),
                 extension="jpg",
                 eager=True):
        """
        - clean_fn: A function that converts the jpeg filename to a string
          frame index. For example, if the jpeg filename is "img_123.jpg", the
          clean_fn should return "123".
        - eager: If True, all the frames are pre-loaded into memory. If False,
          the frames are loaded on demand.
        """
        data_path = Path(data_path)
        self.all_jpgs = sorted(data_path.glob(f"*.{extension}"),
                               key=lambda x: int(clean_fn(x.stem)))
        assert len(self.all_jpgs) > 0, f"No {extension}s found in {data_path}"
        if eager:
            self.frames = [
                cv2.imread(str(x)).astype(np.uint8) for x in self.all_jpgs
            ]
        else:
            self.frames = {}
        self.eager = eager

    def total_num_frames(self):
        return len(self.all_jpgs)

    def get_frame(self, frame_index):
        if self.eager or frame_index in self.frames:
            return {"center_camera_feed": self.frames[frame_index]}
        else:
            self.frames[frame_index] = cv2.imread(
                str(self.all_jpgs[frame_index])).astype(np.uint8)
            return {"center_camera_feed": self.frames[frame_index]}


class JpgDirectorySensor(ImageDirectorySensor):

    def __init__(self, data_path, clean_fn=None, eager=True):
        # pass clean_fn to superclass if it's not None
        kwargs = {}
        if clean_fn is not None:
            kwargs["clean_fn"] = clean_fn
        super().__init__(data_path, extension="jpg", eager=eager, **kwargs)


class PngDirectorySensor(ImageDirectorySensor):

    def __init__(self, data_path, clean_fn=None, eager=True):
        # pass clean_fn to superclass if it's not None
        kwargs = {}
        if clean_fn is not None:
            kwargs["clean_fn"] = clean_fn
        super().__init__(data_path, extension="png", eager=eager, **kwargs)


dataset_readers = {
    "waymo": OfflineWaymoSensorV1_1,
    "argoverse": OfflineWaymoSensorV1_1,
    "MEVA": MEVASensor,
    "visdrone": JpgDirectorySensor,
    "kitti-step": PngDirectorySensor,
}

# variable_resolution is True if the resolution is not the same for all
# scenarios in the dataset.
# resolution is a tuple of (height, width). If variable_resolution is True,
# then the images will be resized to the resolution specified in the tuple for
# inference
dataset_resolutions = {
    "waymo": dict(variable_resolution=False, resolution=(1280, 1920)),
    "argoverse": dict(variable_resolution=False, resolution=(1280, 1920)),
    "MEVA": dict(variable_resolution=False, resolution=(1072, 1920)),
    "kitti-step": dict(variable_resolution=False, resolution=(375, 1242)),
    "visdrone": dict(variable_resolution=True, resolution=(1080, 1920)),
}


def dataset_scenarios_location(dataset):
    """
    Returns the path to the scenarios file for a given dataset.
    """
    json_file = {
        "waymo": "waymo_scenarios.json",
        "argoverse": "argoverse_scenarios.json",
        "MEVA": "MEVA_scenarios.json",
        "visdrone": "visdrone_scenarios.json",
        "kitti-step": "kitti-step_scenarios.json",
    }
    return Path("dataset_scenarios") / json_file[dataset]


def sync_from_google_storage(base_path, path, directory=False):
    """
    Download a file in path in ad-config-search's bucket saved in google cloud
    """
    path = str(path)
    if (Path(base_path) / path).exists():
        return
    print(f"Downloading {path} from google cloud...")
    client = storage.Client.from_service_account_json(
        "erdos-policy-b090169b4a6a.json", project="erdos-policy")
    bucket = client.bucket("ad-config-search")
    blob = bucket.blob(path)
    p = Path(base_path) / path
    if directory:
        to_make = p
    else:
        to_make = p.parent
    to_make.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(p))
