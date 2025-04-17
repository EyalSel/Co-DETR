import numpy as np


def convert_raw_preds_to_edet_format(predictions):
    """
    Convert the raw detection predictions from the model to the EDet format.
    """
    all_preds = []
    for class_id, class_preds in enumerate(predictions):
        # xmin, ymin, xmax, ymax, score -> ymin, xmin, ymax, xmax, score
        class_preds = class_preds[:, [1, 0, 3, 2, 4]]
        all_preds.append(
            np.column_stack([
                -np.ones(len(class_preds)).reshape(-1, 1), class_preds,
                (np.ones(len(class_preds)) * class_id).reshape(-1, 1)
            ]))
    all_preds = np.concatenate(all_preds, axis=0)
    all_preds = all_preds[np.argsort(all_preds[:, 5])[::-1]][:100]
    return all_preds


def edet_format_to_xywh(predictions):
    """
    Convert the EDet format to the xywh format.
    """
    # -1, ymin, xmin, ymax, xmax, score, label -> xmin, ymin, width, height
    return np.column_stack([
        predictions[:, [2, 1]], predictions[:, [4]] - predictions[:, [2]],
        predictions[:, [3]] - predictions[:, [1]]
    ])


def scale_predictions(predictions, previous_resolution, new_resolution):
    """
    Scale the edet format predictions by the scale factor.
    """
    prev_h, prev_w = previous_resolution
    new_h, new_w = new_resolution
    assert np.all(predictions[:,
                              3] <= prev_h), (np.max(predictions[:,
                                                                 3]), prev_h)
    assert np.all(predictions[:,
                              4] <= prev_w), (np.max(predictions[:,
                                                                 4]), prev_w)
    height_scale_factor = new_h / prev_h
    width_scale_factor = new_w / prev_w
    # changing ymin, xmin, ymax, xmax
    predictions[:, [1, 3]] *= height_scale_factor
    predictions[:, [2, 4]] *= width_scale_factor
    return predictions
