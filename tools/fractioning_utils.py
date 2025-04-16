from itertools import product

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import DataContainer


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


class EqualFractions(FractioningSchema):
    """
    Cuts frame into equal pieces (ratio can be a tuple for h and w), and
    supports adding margins to reduce object cropping (margin can be specified
    for h and w dimensions).
    """

    def __init__(self, ratio, margin, with_full_frame=False):
        if isinstance(ratio, tuple):
            assert len(ratio) == 2, ratio
        else:
            ratio = (ratio, ratio)
        self.h_ratio, self.w_ratio = ratio
        if isinstance(margin, tuple):
            assert len(margin) == 2, margin
        else:
            margin = (margin, margin)
        self.h_margin, self.w_margin = margin
        self.with_full_frame = with_full_frame

    def get_split_specs(self, h, w):
        result = []
        for i, j in product(range(self.h_ratio), range(self.w_ratio)):
            ymin = np.clip(int(h / self.h_ratio) * i - self.h_margin, 0, h)
            ymax = np.clip(
                int(h / self.h_ratio) * (i + 1) + self.h_margin, 0, h)
            xmin = np.clip(int(w / self.w_ratio) * j - self.w_margin, 0, w)
            xmax = np.clip(
                int(w / self.w_ratio) * (j + 1) + self.w_margin, 0, w)
            result.append(
                FrameFraction(int(ymin), int(xmin), int(ymax), int(xmax)))
        if self.with_full_frame:
            # Preserves the aspect ratio of the full frame while reducing the
            # area to the size of the crop.
            side_ratio = np.sqrt(self.h_ratio * self.w_ratio)
            (new_h,
             new_w) = (None,
                       None) if side_ratio == 1 else (int(h / side_ratio),
                                                      int(w / side_ratio))
            result.append(FrameFraction(0, 0, h, w, new_h, new_w))
        return result


def run_split_specs_fn(run_fn, split_specs):

    def fn(inp):
        all_results = []
        for split_spec in split_specs:
            # get data from metadata DataContainer
            data = inp["img_metas"][0].data
            img = inp["img"]
            assert len(img) == 1, len(img)
            img = img[0]
            assert img.shape[0] == 1, img.shape
            img = img[0]
            _, og_h, og_w = img.shape
            # crop img
            img = img[:, split_spec.ymin:split_spec.ymax,
                      split_spec.xmin:split_spec.xmax]
            if split_spec.height is not None:
                # resize imgs
                img, w_scale, h_scale = mmcv.imresize(img.numpy().transpose(
                    [1, 2, 0]), (split_spec.width, split_spec.height),
                                                      return_scale=True,
                                                      interpolation='bilinear',
                                                      backend='cv2')
                img = torch.tensor(img.transpose([2, 0, 1]))
                # Reshape
                scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                        dtype=np.float32)
                data[0][0]["scale_factor"] = scale_factor
            data[0][0]["img_shape"] = (img.shape[1], img.shape[2],
                                       img.shape[0])
            # turn metadata back into DataContainer
            d = DataContainer(data, cpu_only=True)
            new_inp = {"img": [img[np.newaxis, :]], "img_metas": [d]}
            result = run_fn(new_inp)
            assert len(result) == 1, len(result)
            result = result[0]
            margin_dist = 3
            # xmin, ymin, xmax, ymax
            for i in range(len(result)):
                result[i][:, 0] += split_spec.xmin
                result[i][:, 1] += split_spec.ymin
                result[i][:, 2] += split_spec.xmin
                result[i][:, 3] += split_spec.ymin
                if split_spec.ymin > 0:
                    result[i] = result[i][result[i][:, 1] > split_spec.ymin +
                                          margin_dist]
                if split_spec.ymax < og_h:
                    result[i] = result[i][result[i][:, 3] < split_spec.ymax -
                                          margin_dist]
                if split_spec.xmin > 0:
                    result[i] = result[i][result[i][:, 0] > split_spec.xmin +
                                          margin_dist]
                if split_spec.xmax < og_w:
                    result[i] = result[i][result[i][:, 2] < split_spec.xmax -
                                          margin_dist]
            all_results.append(result)
        all_preds = []
        for class_id, all_class_preds in enumerate(zip(*all_results)):
            if len(all_class_preds) > 1:
                all_class_preds = np.concatenate(all_class_preds, axis=0)
            else:
                all_class_preds = all_class_preds[0]
            all_preds.append(
                np.concatenate([
                    np.full([len(all_class_preds), 1], -1),
                    all_class_preds,
                    np.full([len(all_class_preds), 1], class_id),
                ],
                               axis=1))
            # breakpoint()
        all_preds = np.concatenate(all_preds, axis=0)
        # breakpoint()
        xmin_ymin_wh_boxes = np.concatenate([
            all_preds[:, [1, 2]], all_preds[:, [3]] - all_preds[:, [1]],
            all_preds[:, [4]] - all_preds[:, [2]]
        ],
                                            axis=1)

        indices = cv2.dnn.NMSBoxes(xmin_ymin_wh_boxes, all_preds[:, 5], 0.3,
                                   0.5)

        all_preds = all_preds[indices]
        all_preds = all_preds[:, [0, 2, 1, 4, 3, 5, 6]]
        all_preds = all_preds[np.argsort(all_preds[:, 5])[::-1]][:100]

        # The commented code below converts the results into a format that can
        # be saved as an image. If this is uncommented, the uncomment
        # show_preds at the bottom of main().

        # x = [[] for _ in range(80)]
        # for entry in all_preds:
        #     x[int(entry[6])].append(entry[[2, 1, 4, 3, 5]])
        # x = [np.array(y) for y in x]
        # x = [y.reshape(0, 5) if len(y) == 0 else y for y in x]
        # x = [x]
        # return x
        return all_preds

    return fn
