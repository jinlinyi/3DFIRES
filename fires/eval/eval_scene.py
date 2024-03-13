import os
import os.path as osp
import pickle
from collections import defaultdict

import numpy as np
import scipy.spatial
import trimesh
from loguru import logger
from tqdm import tqdm

from fires.utils.geometry_utils import project_ndc_depth


def distance_p2p(
    points_src,
    points_tgt,
):
    """Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """

    kdtree = scipy.spatial.KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)
    return dist


def rowData2latex(
    data,
    bold_col=None,
):
    strs = []
    if bold_col is None:
        bold_col = [False for _ in data]
    for col, bold in zip(data, bold_col):
        if bold:
            strs.append("\\textbf{" + f"{col:.2f}" + "}")
        else:
            strs.append(f"{col:.2f}")
    return " & ".join(strs)


class VisibilityCategory:
    def __init__(self):
        self.categories = [
            "all-hidden",
            "any-visible",
        ]

    def classify(self, visibility, valid_mask):
        category_masks = {}
        for cat in self.categories:
            if cat == "all-hidden":
                category_masks[cat] = (valid_mask.sum(1) >= 1) & (
                    visibility.sum(1) == 0
                )
            elif cat == "any-visible":
                category_masks[cat] = (valid_mask.sum(1) >= 1) & (visibility.sum(1) > 0)
            else:
                raise NotImplementedError()
        return category_masks


## This only works for one instance.
class EvaluateInstance:
    def __init__(self, uuid_name, thresholds):
        self.thresholds = thresholds
        self.classifier = VisibilityCategory()
        return

    def evaluate(self, pred_dict, gt_dict):
        gt_pts = gt_dict["scene_pcd"].cpu().numpy()
        pred_pts = pred_dict["pcd_world"].cpu().numpy()

        gt_visibility = gt_dict["scene_visibility"]
        pred_visibility = pred_dict["scene_visibility"]

        gt_valid_mask = gt_dict["scene_valid_mask"]
        pred_valid_mask = pred_dict["scene_valid_mask"]

        gt_category_masks = self.classifier.classify(gt_visibility, gt_valid_mask)
        pred_category_masks = self.classifier.classify(pred_visibility, pred_valid_mask)
        total_num_pts = np.sum(np.any(gt_dict["scene_valid_mask"].numpy(), axis=1))

        pts_ratio = {}
        for key in gt_category_masks:
            ratio = (gt_category_masks[key].sum() / total_num_pts).numpy()
            pts_ratio[key] = ratio

        eval_result = {}
        for key in gt_category_masks.keys():
            eval_result[key] = self.calculate_pr(
                gt_pts[gt_category_masks[key]], pred_pts[pred_category_masks[key]]
            )
        eval_result["All"] = self.calculate_pr_combined(
            gt_pts,
            pred_pts,
            gt_category_masks,
            pred_category_masks,
            ["any-visible", "all-hidden"],
        )
        pts_ratio["All"] = 1
        return eval_result, pts_ratio

    def calculate_pr(self, gt_points, pred_points):
        thresholds = self.thresholds
        precision = []
        recall = []
        fscore = []
        pred2gt = distance_p2p(points_src=pred_points, points_tgt=gt_points)
        gt2pred = distance_p2p(points_src=gt_points, points_tgt=pred_points)

        for tx, threshold in enumerate(thresholds):
            # self.calculate_prAT(mesh=gt_mesh, gt_points=gt_points,
            #                     pred_points=pred_points,
            #                     threshold=threshold)
            precisionTx = (pred2gt < threshold).mean()
            recallTx = (gt2pred < threshold).mean()
            fscoreTx = 2 * (precisionTx * recallTx) / (precisionTx + recallTx + 1e-4)
            precision.append(precisionTx)
            recall.append(recallTx)
            fscore.append(fscoreTx)

        precision = np.stack(precision)
        recall = np.stack(recall)
        fscore = np.stack(fscore)

        eval_result = Struct(
            precision=precision,
            recall=recall,
            fscore=fscore,
            thresholds=self.thresholds,
        )

        return eval_result

    def calculate_pr_combined(
        self,
        gt_points_all,
        pred_points_all,
        gt_category_masks,
        pred_category_masks,
        keys,
    ):
        thresholds = self.thresholds
        precision = defaultdict(list)
        recall = defaultdict(list)
        for key in keys:
            gt_points = gt_points_all[gt_category_masks[key]]
            pred_points = pred_points_all[pred_category_masks[key]]

            pred2gt = distance_p2p(points_src=pred_points, points_tgt=gt_points)
            gt2pred = distance_p2p(points_src=gt_points, points_tgt=pred_points)
            for tx, threshold in enumerate(thresholds):
                precision[tx].extend(pred2gt < threshold)
                recall[tx].extend(gt2pred < threshold)

        precision_overall = []
        recall_overall = []
        fscore_overall = []
        for tx, threshold in enumerate(thresholds):
            p_m = np.array(precision[tx]).mean()
            r_m = np.array(recall[tx]).mean()
            fscore = 2 * (p_m * r_m) / max(p_m + r_m, 1e-4)
            precision_overall.append(p_m)
            recall_overall.append(r_m)
            fscore_overall.append(fscore)

        precision = np.stack(precision_overall)
        recall = np.stack(recall_overall)
        fscore = np.stack(fscore_overall)
        eval_result = Struct(
            precision=precision,
            recall=recall,
            fscore=fscore,
            thresholds=self.thresholds,
        )

        return eval_result


class Struct:
    def __init__(self, **kwargs):
        self.struct_keys = []
        for key, val in kwargs.items():
            setattr(self, key, val)
            self.struct_keys.append(key)

    def keys(
        self,
    ):
        return self.struct_keys


## This will carry out the complete evaluation
class FscoreRepository:
    def __init__(
        self,
    ):
        self.scene_pr_thresholds = [0.05, 0.1, 0.2, 0.5]
        self.eval_result_lst = defaultdict(list)
        self.pts_ratio_lst = defaultdict(list)
        self.count = 0
        return

    def eval_single(self, uuid, predictions, gt_dict):
        eval_result, pts_ratio = EvaluateInstance(
            uuid, self.scene_pr_thresholds
        ).evaluate(predictions, gt_dict)

        if eval_result is not None:
            for key in eval_result:
                self.eval_result_lst[key].append(eval_result[key])

        if pts_ratio is not None:
            for key in pts_ratio:
                self.pts_ratio_lst[key].append(pts_ratio[key])

        self.count += 1

    def summary(self):
        dataframe = {}

        for key in self.eval_result_lst.keys():
            precision = []
            recall = []
            fscore = []
            for eval_result in self.eval_result_lst[key]:
                precision.append(eval_result.precision)
                recall.append(eval_result.recall)
                fscore.append(eval_result.fscore)

            precision = np.stack(precision)
            recall = np.stack(recall)
            fscore = np.stack(fscore)

            precision = np.mean(precision, axis=0)
            recall = np.mean(recall, axis=0)
            fscore = np.mean(fscore, axis=0)
            combined_result = Struct(precision=precision, recall=recall, fscore=fscore)

            logger.info("compute PR for all")

            threshold_str = rowData2latex(np.array(self.scene_pr_thresholds))
            precision_str = rowData2latex(precision * 100)
            recall_str = rowData2latex(recall * 100)
            fscore_str = rowData2latex(fscore * 100)

            logger.info(f"Metric {key}")

            logger.info(f"Threshold {threshold_str}")
            logger.info(f"Precision  {precision_str} ")
            logger.info(f"Recall  {recall_str} ")
            logger.info(f"Fscore  {fscore_str} ")
            logger.info(f"Ratio  {np.mean(self.pts_ratio_lst[key])*100 :.1f}% ")

            dataframe[key] = {
                "num_iter": self.count,
                "precision": precision,
                "recall": recall,
                "fscore": fscore,
                "threshold": np.array(self.scene_pr_thresholds),
                "ratio": np.mean(self.pts_ratio_lst[key]),
            }

        return self.eval_result_lst, combined_result


class ConsistencyRepository:
    """
    Recall:  For every ground truth point find the closest predicted point, as well as its index in the predicted ray.
    """

    def __init__(self):
        super().__init__()
        self.consistency = {}
        self.threshold = thresholds = [0.05, 0.1, 0.2, 0.5]
        return

    def eval_single(self, idx, predictions, dataset_dict):
        for camid in range(len(dataset_dict["camera_torch"])):
            p_img_id = (
                predictions["imgid"]
                if "imgid" in predictions
                else predictions["img_ids"]
            )
            pcd_others = predictions["pcd_world"][p_img_id != camid]
            pcd_ndc = project_ndc_depth(
                pcd_others,
                dataset_dict["camera_torch"][camid].cpu(),
                znear=None,
                zfar=None,
            )
            valid = (
                (pcd_ndc[..., 0] < 1)
                & (pcd_ndc[..., 1] < 1)
                & (pcd_ndc[..., 2] < 8)
                & (pcd_ndc[..., 0] > -1)
                & (pcd_ndc[..., 1] > -1)
                & (pcd_ndc[..., 2] > 0)
            )
            pcd_others = pcd_others[valid]

            pcd_ref = predictions["pcd_world"][p_img_id == camid]

            ref2other = distance_p2p(points_src=pcd_others, points_tgt=pcd_ref)

            self.get_consistency(idx, ref2other)

    def get_consistency(self, idx, distances):
        # for all the other points inside the reference frustum,
        # calculate if the reference frustum predicts something close to it.
        # ratio: predictions from the other view that is consistent / predictions from the other view

        rtn_array = np.zeros((2, len(self.threshold)))
        for thid, th in enumerate(self.threshold):
            rtn_array[0, thid] = (distances < th).sum()
            rtn_array[1, thid] = len(distances)
        if idx not in self.consistency:
            self.consistency[idx] = rtn_array
        else:
            self.consistency[idx] = self.consistency[idx] + rtn_array
        return True

    def summary(self):
        good_ratio_individual = []
        for idx, array in self.consistency.items():
            good_ratio_individual.append(array[0] / array[1])

        individual_mean = np.mean(np.array(good_ratio_individual), axis=0)
        logger.info("scene_consistency")
        logger.info(" & ".join([f"{m :.2f}" for m in self.threshold]) + " \\\\")
        logger.info(" & ".join([f"{m*100 :.2f}" for m in individual_mean]) + " \\\\")
        dataframe = {
            "scene_consistency": {
                "num_iter": len(self.consistency.items()),
                "threshold": self.threshold,
                "consistency": individual_mean,
            }
        }

        return dataframe
