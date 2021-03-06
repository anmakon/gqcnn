import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image, ImageDraw, ImageFont, ImageColor
import json
from autolab_core import YamlConfig, Logger, BinaryClassificationResult, Point
from collections import defaultdict
import pandas as pd
import csv

import os
import argparse
import sklearn

from gqcnn.model import get_gqcnn_model
from path_settings import DATA_PATH, EXPER_PATH
from gqcnn.grasping import Grasp2D

"""
This is a script to conduct a detailed analysis of a Cornell- or Dexnet subset.
The influence of noise or different grasping depths on the GQCNN can be analysed.

"""


class GQCNN_Analyse():

    def __init__(self, verbose=True, plot_backend="pdf", model="DexNet", dset='DexNet'):

        self.metric_thresh = 0.002  # Change metric threshold here if needed!
        self.verbose = verbose
        plt.switch_backend(plot_backend)
        self.num_images = 100  # Amount of images for plotting. Set to None if you want to plot all images
        self.model = model
        self.dset = dset

    @staticmethod
    def scale(data, x_min=0, x_max=255):
        """Scale a depth image in order to visualise it as a greyscale
        image. Fixed scaling from [0.6,0.75] to [0,255]."""
        data_fl = data.flatten()
        scaled = np.interp(data_fl, (0.6, 0.8), (x_min, x_max))
        integ = scaled.astype(np.uint8)
        integ.resize((32, 32))
        return integ

    def _plot_grasp(self, image_arr, width, results, j, plt_results=True, noise_arr=None, depth_arr=None,
                    perturb_arr=None):
        """ Creating images with the grasps.
        Adding text with the added noise or rotation/translation.
        Visualising the grasp position and width"""

        data = self.scale(image_arr[:, :, 0])
        image = Image.fromarray(data).convert('RGB')
        font = ImageFont.truetype(font=DATA_PATH + "arial_narrow_7.ttf", size=20)
        draw = ImageDraw.Draw(image)
        if results.labels[j] == 0 and plt_results:
            filling = (250, 0, 0, 128)
        else:
            filling = (0, 100, 0, 128)
        # blue = (0, 0, 250, 128)
        # if self.model == 'DexNet' and self.dset == 'Cornell':
        #     draw.line([16 - 6 - 5, 16, 16 - 6, 16], fill=filling)  # Grasp line
        #     draw.line([16 + 6, 16, 16 + 6 + 5, 16], fill=filling)  # Grasp line
        #
        #     draw.line([16 - 6, 13, 16 - 6, 19], fill=filling)  # Vertical lines for end of grasp
        #     draw.line([16 + 6, 13, 16 + 6, 19], fill=filling)
        # else:
        #     draw.line([16 - width / 2 - 5, 16, 16 - width / 2, 16], fill=blue)  # Grasp line
        #     draw.line([16 + width / 2, 16, 16 + width / 2 + 5, 16], fill=blue)  # Grasp line
        #     filling = blue
        # # if self.dset == 'Cornell':
        # draw.line([16 - width / 2, 13, 16 - width / 2, 19], fill=blue)  # Vertical lines for end of real Cornell grasp
        # draw.line([16 + width / 2, 13, 16 + width / 2, 19], fill=blue)

        # Draw grasp point

        draw.line([16, 16, 16, 16], fill=filling)

        image = image.resize((300, 300), resample=Image.NEAREST)
        # Add prediction and label
        if plt_results:
            draw2 = ImageDraw.Draw(image)
            draw2.text((3, 3), "Pred: %.3f; Label: %.1f" % (results.pred_probs[j], results.labels[j]), fill=50,
                       font=font)
            if noise_arr is not None:
                draw2.text((3, 18), "Added noise: %.4f; Added tilting: %.3f" % (noise_arr[j, 0], noise_arr[j, 1]),
                           fill=50, font=font)
            if depth_arr is not None:
                if depth_arr[j] == -1:
                    draw2.text((3, 18), "Original depth", fill=50, font=font)
                else:
                    draw2.text((3, 18), "Realtive depth %.2f" % depth_arr[j], fill=50, font=font)
            if perturb_arr is not None:
                ypos = 18
                print(perturb_arr[j, 1])
                if perturb_arr[j, 0] != 0:
                    draw2.text((3, ypos), "Grasp rotation: %.1f degree" % perturb_arr[j, 0], fill=50, font=font)
                    ypos += 18
                if perturb_arr[j, 1] != 0:
                    draw2.text((3, ypos), "Grasp translationx: %.1f pixel" % perturb_arr[j, 1], fill=50, font=font)
                    ypos += 18
                if perturb_arr[j, 2] != 0:
                    draw2.text((3, ypos), "Grasp translationy: %.1f pixel" % perturb_arr[j, 2], fill=50, font=font)
                    ypos += 18
                if perturb_arr[j, 3] != 0:
                    draw2.text((3, ypos), "Grasp scale depth: %.1f" % perturb_arr[j, 3], fill=50, font=font)
                    ypos += 18
                if perturb_arr[j, 4] != 0:
                    draw2.text((3, ypos), "Grasp scale x: %.1f" % perturb_arr[j, 4], fill=50, font=font)
        return image

    def _plot_histograms(self, predictions, labels, savestring, output_dir):
        """ Plot histograms with the absolute prediction errors.
        Can be done for positive and negative grasps, depending
        on the given input."""
        pos_errors, neg_errors = self._calculate_prediction_errors(predictions, labels)
        binwidth = 0.02
        if len(pos_errors) > 0:
            plt.hist(pos_errors, bins=np.arange(0, 1 + binwidth, binwidth))
            axes = plt.gca()
            y0, y1 = axes.get_ylim()
            plt.axvline(x=0.5, color='r')
            plt.text(0.1, y1 / 2, 'Positive prediction', color='r')
            plt.text(0.6, y1 / 2, 'Negative prediction', color='r')
            plt.xlabel("Absolute Prediction Error", fontsize=14)
            plt.title("Error on successful grasps", fontsize=18)
            plt.savefig(output_dir + "/err_pos_" + savestring + ".png")
            plt.close()

        if len(neg_errors) > 0:
            plt.hist(neg_errors, bins=np.arange(0, 1 + binwidth, binwidth))
            axes = plt.gca()
            y0, y1 = axes.get_ylim()
            plt.axvline(x=0.5, color='r')
            plt.text(0.1, y1 / 2, 'Positive prediction', color='r')
            plt.text(0.6, y1 / 2, 'Negative prediction', color='r')
            plt.xlabel("Absolute Prediction Error", fontsize=14)
            plt.title("Error on unsuccessful grasps", fontsize=18)
            plt.savefig(output_dir + "/err_neg_" + savestring + ".png")
            plt.close()

            # plt.rc('axes',edgecolor='w')
            # plt.rc('text',color='w')
            # plt.rc(('xtick','ytick'),c='w')
            #
            # plt.hist(pos_errors,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
            # plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
            # plt.title("Error on successful grasps",fontsize=18)
            # plt.savefig(output_dir+"/err_pos_"+savestring+".png",transparent=True)
            # plt.close()
            #
            # plt.hist(neg_errors,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
            # plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
            # plt.title("Error on unsuccessful grasps",fontsize=18)
            # plt.savefig(output_dir+"/err_neg_"+savestring+".png",transparent=True)
            # plt.close()

    def _calculate_prediction_errors(self, predictions, labels):
        pos_ind = np.where(labels == 1)
        neg_ind = np.where(labels == 0)
        pos_prediction_errors = np.abs(labels[pos_ind] - predictions[pos_ind])
        neg_prediction_errors = np.abs(labels[neg_ind] - predictions[neg_ind])
        return pos_prediction_errors, neg_prediction_errors

    def _plot_grasp_perturbations(self, degrees, accuracies, output_dir, mode):
        """Plot the grasp pertrubation against the classification accuracies
        for multiple grasps, e.g. a whole dataset. This is usually done to
        see the sensitivity of a whole dataset or a sub-dataset to perturbations"""
        if mode == 'rotation':
            unit = 'deg'
        elif mode == 'translation':
            unit = 'pixel'
        plt.plot(degrees, accuracies, '-x', linewidth=2, markersize=10)
        plt.grid(color=(0.686, 0.667, 0.667), linestyle='--')
        plt.title(
            'Classification accuracy for ' + self.dset + ' \n with grasp ' + mode + '. Model trained on ' + self.model,
            fontsize=16)
        plt.xlabel("Grasp " + mode + " perturbation [" + unit + "]", fontsize=12)
        plt.ylabel("Classification accuracy [%]", fontsize=12)
        plt.ylim((0, 102))
        plt.savefig(output_dir + "/Grasp_" + mode + "_accuracy.png")
        plt.close()

    def _plot_single_grasp_perturbations(self, degrees, pred_err, output_dir, mode):
        """ Plot the grasp perturbation against the absolute prediction error.
        This is done for single grasps, usually if you have one grasp and want
        to visualise the prediction over rotations/translations."""
        if mode == 'rotation':
            unit = 'deg'
        elif mode == 'translation':
            unit = 'pixel'
        elif mode == 'translationy':
            unit = 'pixel'
        plt.plot(degrees, pred_err, '-x', linewidth=2, markersize=10)
        plt.grid(color=(0.686, 0.667, 0.667), linestyle='--')
        plt.title(
            'Prediction errors for positive ' + self.dset + ' grasps \n with ' + mode + '. Model trained on ' + self.model,
            fontsize=16)
        plt.xlabel("Grasp " + mode + " perturbation [" + unit + "]", fontsize=12)
        plt.ylabel("Absolute prediction error", fontsize=12)
        plt.ylim((0, 1.1))
        axes = plt.gca()
        x0, x1 = axes.get_xlim()
        horizontal = np.array([0.5 for i in range(len(degrees))])
        plt.text(x0 / 2, 0.25, 'Positive prediction', color='r')
        plt.text(x0 / 2, 0.75, 'Negative prediction', color='r')
        plt.plot(degrees, horizontal, 'r--')
        plt.savefig(output_dir + "/Grasp_" + mode + "_err.png")
        plt.close()

    def visualise(self, model_dir, output_dir, data_dir):

        # Determine model name
        model_name = ""
        model_root = model_dir
        while model_name == "" and model_root != "":
            model_root, model_name = os.path.split(model_root)

        output_dir = os.path.join(output_dir, "Visualisation/")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Read in model config.
        model_config_filename = os.path.join(model_dir, "config.json")
        with open(model_config_filename) as data_file:
            model_config = json.load(data_file)

        # Set up logger
        self.logger = Logger.get_logger(self.__class__.__name__,
                                        log_file=os.path.join(
                                            output_dir, "analysis.log"),
                                        silence=(not self.verbose),
                                        global_log_file=self.verbose)

        self.logger.info("Analyzing model %s" % (model_name))
        self.logger.info("Saving output to %s" % (output_dir))
        mixture = False

        if "Cornell" in model_dir:
            model_name = "Cornell"
        elif "DexNet" in model_dir:
            model_name = "DexNet"

        if "Cornell" in data_dir:
            data_name = "Cornell"
        elif "DexNet" in data_dir:
            data_name = "DexNet"
        elif "Both" in data_dir:
            data_name = "mixed"
            mixture = True

        # Load model.
        self.logger.info("Loading model %s" % (model_dir))
        log_file = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = handler.baseFilename
        gqcnn = get_gqcnn_model(verbose=self.verbose).load(
            model_dir, verbose=self.verbose, log_file=log_file)
        gqcnn.open_session()
        gripper_mode = gqcnn.gripper_mode
        angular_bins = gqcnn.angular_bins

        if mixture:
            image_arr, pose_arr, labels, width_arr, file_arr, obj_arr, identity_arr = self._read_data(data_dir,
                                                                                                      mixture=True)
        else:
            image_arr, pose_arr, labels, width_arr, file_arr, obj_arr = self._read_data(data_dir)
        print("Object arr: ", obj_arr)
        # Predict outcomes
        predictions = gqcnn.predict(image_arr, pose_arr)

        if predictions.shape[1] == 1:
            print("Only 1 image given. No t-SNE analysis of network possible")
        else:
            # Setting colors and labels
            color = []
            monotone = False
            if mixture:
                for label, identity in zip(labels, identity_arr):
                    if identity == 0:
                        # Cornell
                        if label == 0:
                            # negative
                            color.append('#FF8000')
                        else:
                            # positive
                            color.append('#2D702F')
                        # DexNet
                        if label == 0:
                            # negative
                            color.append('#FF0404')
                        else:
                            # positive
                            color.append('#23C328')
                if len(np.unique(labels)) == 1:
                    monotone = True
                    if labels[0] == 0:
                        data_name += " negatives"
                        pop_a = mpatches.Patch(color='#FF8000', label='Negative Cornell')
                        pop_b = mpatches.Patch(color='#FF0404', label='Negative DexNet')
                    else:
                        data_name += " positives"
                        pop_a = mpatches.Patch(color='#2D702F', label='Positive Cornell')
                        pop_b = mpatches.Patch(color='#23C328', label='Positive DexNet')
                else:
                    pop_a = mpatches.Patch(color='#FF8000', label='Negative Cornell')
                    pop_b = mpatches.Patch(color='#FF0404', label='Negative DexNet')
                    pop_c = mpatches.Patch(color='#2D702F', label='Positive Cornell')
                    pop_d = mpatches.Patch(color='#23C328', label='Positive DexNet')
            else:
                color = ['r' if truth == 0 else 'g' for truth in labels]
                pop_a = mpatches.Patch(color='r', label='Negative grasp')
                pop_b = mpatches.Patch(color='g', label='Positive grasp')

            # t-SNE
            tsne_out = sklearn.manifold.TSNE(n_components=2).fit_transform(predictions)
            plt.scatter(tsne_out[:, 0], tsne_out[:, 1], marker='o', c=color)
            if mixture and not monotone:
                plt.legend(handles=[pop_a, pop_b, pop_c, pop_d])
            else:
                plt.legend(handles=[pop_a, pop_b])
            plt.title("TSNE output of %s data on a GQCNN trained on %s" % (data_name, model_name))
            plt.savefig(output_dir + "/" + model_name + "_model_" + data_name + "_data_TSNE.png")
            plt.close()

            # PCA
            pca_out = sklearn.decomposition.PCA(n_components=2).fit_transform(predictions)
            plt.scatter(pca_out[:, 0], pca_out[:, 1], marker='o', c=color)
            plt.title("PCA output of %s data on a GQCNN trained on %s" % (data_name, model_name))
            if mixture and not monotone:
                plt.legend(handles=[pop_a, pop_b, pop_c, pop_d])
            else:
                plt.legend(handles=[pop_a, pop_b])
            plt.savefig(output_dir + "/" + model_name + "_model_" + data_name + "_data_PCA.png")
            plt.close()

    def run_analysis(self, model_dir, output_dir, data_dir, noise_analysis, depth_analysis, perturb_analysis,
                     single_analysis):

        # Determine model name
        model_name = ""
        model_root = model_dir
        while model_name == "" and model_root != "":
            model_root, model_name = os.path.split(model_root)

        # Store Noise and Depth investigation in their corresponding directories
        if noise_analysis:
            output_dir = os.path.join(output_dir, "Noise_Comparison/")
        if depth_analysis:
            output_dir = os.path.join(output_dir, "Depth_Comparison/")
        if perturb_analysis:
            output_dir = os.path.join(output_dir, "Perturbation_Analysis/")
        if single_analysis:
            output_dir = os.path.join(output_dir, "Single_Analysis/")

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Set up logger.
        self.logger = Logger.get_logger(self.__class__.__name__,
                                        log_file=os.path.join(
                                            output_dir, "analysis.log"),
                                        silence=(not self.verbose),
                                        global_log_file=self.verbose)

        self.logger.info("Analyzing model %s" % (model_name))
        self.logger.info("Saving output to %s" % (output_dir))

        # Run predictions
        result = self._run_prediction(model_dir, output_dir, data_dir, noise_analysis, depth_analysis, perturb_analysis,
                                      single_analysis)

    def _export_object_analysis(self, true_pos, false_neg, false_pos, true_neg, output_dir):
        with open(output_dir + "/object_analysis.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Object name', 'True positive', 'True negative', 'False positive', 'False negative'])
            for key in true_pos.keys():
                writer.writerow([key, true_pos[key], true_neg[key], false_pos[key], false_neg[key]])
        return None

    def _run_prediction(self, model_dir, model_output_dir, data_dir, noise_analysis, depth_analysis, perturb_analysis,
                        single_analysis):
        """Predict the outcome of the file for a single model."""

        # Read in model config.
        model_config_filename = os.path.join(model_dir, "config.json")
        with open(model_config_filename) as data_file:
            model_config = json.load(data_file)

        # Load model.
        self.logger.info("Loading model %s" % (model_dir))
        log_file = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = handler.baseFilename
        gqcnn = get_gqcnn_model(verbose=self.verbose).load(
            model_dir, verbose=self.verbose, log_file=log_file)
        gqcnn.open_session()
        gripper_mode = gqcnn.gripper_mode
        angular_bins = gqcnn.angular_bins

        # Load data
        if noise_analysis:
            image_arr, pose_arr, labels, width_arr, file_arr, noise_arr = self._read_data(data_dir, noise=True)
        elif depth_analysis:
            image_arr, pose_arr, labels, width_arr, file_arr, depth_arr = self._read_data(data_dir, depth=True)
        elif perturb_analysis:
            image_arr, pose_arr, labels, width_arr, file_arr, perturb_arr = self._read_data(data_dir, perturb=True)
        elif single_analysis:
            image_arr, pose_arr, labels, width_arr, file_arr, perturb_arr = self._read_data(data_dir, perturb=True)
        else:
            image_arr, pose_arr, labels, width_arr, file_arr, obj_arr = self._read_data(data_dir)
        # Predict outcomes
        predictions = gqcnn.predict(image_arr, pose_arr)
        gqcnn.close_session()
        results = BinaryClassificationResult(predictions[:, 1], labels)

        # Log the results
        if noise_analysis:
            # Analyse the error rates in regard to the noise levels of the images
            noise_levels = np.unique(noise_arr)
            levels = len(noise_levels)
            for current_noise in noise_levels:
                pred = predictions[noise_arr[:, 0] == current_noise]
                lab = labels[noise_arr[:, 0] == current_noise]
                res = BinaryClassificationResult(pred[:, 1], lab)
                self._plot_histograms(pred[:, 1], lab, str(current_noise), model_output_dir)
                self.logger.info("Noise: %.4f Model %s error rate: %.3f" %
                                 (current_noise, model_dir, res.error_rate))
                self.logger.info("Noise: %.4f Model %s loss: %.3f" %
                                 (current_noise, model_dir, res.cross_entropy_loss))
        elif depth_analysis:
            # Analyse the error rates in regard to the grasping depth in the images
            depth_levels = np.unique(depth_arr)
            levels = len(depth_levels)
            for current_depth in depth_levels:
                if current_depth == -1:
                    depth_mode = 'original'
                else:
                    depth_mode = 'relative %.2f' % (current_depth)
                pred = predictions[depth_arr == current_depth]
                lab = labels[depth_arr == current_depth]
                res = BinaryClassificationResult(pred[:, 1], lab)
                self._plot_histograms(pred[:, 1], lab, depth_mode, model_output_dir)
                self.logger.info("Depth %s Model %s error rate: %.3f" %
                                 (depth_mode, model_dir, res.error_rate))
                self.logger.info("Depth: %s Model %s loss: %.3f" %
                                 (depth_mode, model_dir, res.cross_entropy_loss))
        elif perturb_analysis:
            # Analyse the error rates in regard to the grasping perturb in the images
            perturb_levels = np.unique(perturb_arr)
            print("Perturb levels: ", perturb_levels)
            _rot = len(np.unique(perturb_arr[:, 0]))
            _trans = len(np.unique(perturb_arr[:, 1]))
            try:
                _transy = len(np.unique(perturb_arr[:, 2]))
            except:
                _transy = 0
                print("No translation in y included")
            if _rot >= 2 and _trans <= 1 and _transy <= 1:
                perturbation = 'rotation'
                perturb_unit = 'deg'
                index = 0
            elif _rot <= 1 and _trans >= 2 and _transy <= 1:
                perturbation = 'translation'
                perturb_unit = 'pixel'
                index = 1
            elif _rot <= 1 and _trans <= 1 and _transy >= 2:
                perturbation = 'translationy'
                perturb_unit = 'pixel'
                index = 2
            else:
                raise ValueError(
                    "Perturbation array includes at least two different perturbation types. Can't be handled. Abort.")
                return None
            levels = len(perturb_levels)
            accuracies = []
            for current_perturb in perturb_levels:
                pred = predictions[perturb_arr[:, index] == current_perturb]
                lab = labels[perturb_arr[:, index] == current_perturb]
                res = BinaryClassificationResult(pred[:, 1], lab)
                perturb_mode = perturbation + ' %.0f ' % (current_perturb) + perturb_unit
                self._plot_histograms(pred[:, 1], lab, perturbation + '_%.0f_' % (current_perturb) + perturb_unit,
                                      model_output_dir)

                self.logger.info("Grasp %s Model %s error rate: %.3f" %
                                 (perturb_mode, model_dir, res.error_rate))
                accuracies.append(100 - res.error_rate)
                self.logger.info("Grasp %s Model %s loss: %.3f" %
                                 (perturb_mode, model_dir, res.cross_entropy_loss))
            self._plot_grasp_perturbations(perturb_levels, accuracies, model_output_dir, perturbation)
        elif single_analysis:
            # Analyse the error rates in regard to the grasping perturb in the images
            perturb_levels = np.unique(perturb_arr)
            _rot = np.count_nonzero(perturb_arr[:, 0])
            _trans = np.count_nonzero(perturb_arr[:, 1])
            _transy = np.count_nonzero(perturb_arr[:, 2])
            _scalez = np.count_nonzero(perturb_arr[:, 3])
            _scalex = np.count_nonzero(perturb_arr[:, 4])
            if _rot >= 1 and _trans == 0 and _transy == 0 and _scalez == 0 and _scalex == 0:
                index = 0
                perturbation = 'rotation'
            elif _rot == 0 and _trans >= 1 and _transy == 0 and _scalez == 0 and _scalex == 0:
                perturbation = 'translation'
                index = 1
            elif _rot == 0 and _trans == 0 and _transy >= 1 and _scalez == 0 and _scalex == 0:
                perturbation = 'translationy'
                index = 2
            elif _rot == 0 and _trans == 0 and _transy == 0 and _scalez >= 1 and _scalex == 0:
                perturbation = 'scale_height'
                index = 3
            elif _rot == 0 and _trans == 0 and _transy == 0 and _scalez == 0 and _scalex >= 1:
                perturbation = 'scalex'
                index = 4
            else:
                perturbation = 'mixed'
                index = 5
            # Create new output dir for single file and perturbation mode
            print(len(perturb_arr))
            if len(perturb_arr) == 1:
                print("New output direction is: ", model_output_dir)
            else:
                model_output_dir = os.path.join(model_output_dir,
                                                str(file_arr[0][0]) + '_' + str(file_arr[0][1]) + '_' + perturbation)
                print("New output direction is: ", model_output_dir)
            if not os.path.exists(model_output_dir):
                os.mkdir(model_output_dir)
            # Set up new logger.
            self.logger = Logger.get_logger(self.__class__.__name__,
                                            log_file=os.path.join(
                                                model_output_dir, "analysis.log"),
                                            silence=(not self.verbose),
                                            global_log_file=self.verbose)
            levels = len(perturb_arr)
            abs_pred_errors = []
            if levels == 1:
                self.logger.info(
                    "Mixed perturbation. Translationx %.1f, Translationy %.1f, "
                    "Rotation %.1f, Scale_height %.1f, Scale x %.1f" %
                    (perturb_arr[0][1], perturb_arr[0][2], perturb_arr[0][0], perturb_arr[0][3], perturb_arr[0][4]))
                pred = predictions
                lab = labels
                res = BinaryClassificationResult(pred[:, 1], lab)
                self.logger.info("Grasp %s Model %s prediction: %.3f" %
                                 (perturbation, model_dir, pred[:, 1]))
                self.logger.info("Grasp %s Model %s error rate: %.3f" %
                                 (perturbation, model_dir, res.error_rate))
                self.logger.info("Grasp %s Model %s loss: %.3f" %
                                 (perturbation, model_dir, res.cross_entropy_loss))

            else:
                for current_perturb in perturb_levels:
                    pred = predictions[perturb_arr[:, index] == current_perturb]
                    lab = labels[perturb_arr[:, index] == current_perturb]
                    res = BinaryClassificationResult(pred[:, 1], lab)

                    if perturbation == 'rotation':
                        perturb_mode = 'rotation %.0f deg' % (current_perturb)
                    elif perturbation == 'translation':
                        perturb_mode = 'translation in x %.0f pixel' % (current_perturb)
                    elif perturbation == 'translationy':
                        perturb_mode = 'translation in y %.0f pixel' % (current_perturb)
                    elif perturbation == 'scale_height':
                        perturb_mode = 'scaling depth by %.0f' % (current_perturb)
                    elif perturbation == 'scalex':
                        perturb_mode = 'scaling x by %.0f' % (current_perturb)
                    pos_errors, neg_errors = self._calculate_prediction_errors(pred[:, 1], lab)
                    # Only append positive errors if grasp was positive.
                    if pos_errors:
                        abs_pred_errors.append(pos_errors)
                    self.logger.info("Grasp %s Model %s prediction: %.3f" %
                                     (perturb_mode, model_dir, pred[:, 1]))
                    self.logger.info("Grasp %s Model %s error rate: %.3f" %
                                     (perturb_mode, model_dir, res.error_rate))
                    self.logger.info("Grasp %s Model %s loss: %.3f" %
                                     (perturb_mode, model_dir, res.cross_entropy_loss))
                if pos_errors:
                    self._plot_single_grasp_perturbations(perturb_levels, abs_pred_errors, model_output_dir,
                                                          perturbation)
        else:
            levels = 1
            self._plot_histograms(predictions[:, 1], labels, '', model_output_dir)
            self.logger.info("Model %s error rate: %.3f" %
                             (model_dir, results.error_rate))
            self.logger.info("Model %s loss: %.3f" %
                             (model_dir, results.cross_entropy_loss))

            if obj_arr is not None and 'Cornell' in data_dir:
                unique = np.unique(obj_arr).tolist()
                object_label = pd.read_csv(DATA_PATH + "Cornell/original/z.txt", sep=" ", header=None,
                                           usecols=[1, 2]).drop_duplicates().to_numpy()
                true_pos = dict()
                false_neg = dict()
                false_pos = dict()
                true_neg = dict()
                for obj in unique:
                    obj = int(obj)
                    true_pos[object_label[obj, 1]] = 0
                    false_pos[object_label[obj, 1]] = 0
                    true_neg[object_label[obj, 1]] = 0
                    false_neg[object_label[obj, 1]] = 0

                for obj, pred, label in zip(obj_arr, predictions[:, 1], labels):
                    if label == 1 and pred >= 0.5:
                        true_pos[object_label[obj, 1]] += 1
                    elif label == 1 and pred < 0.5:
                        false_neg[object_label[obj, 1]] += 1
                    elif label == 0 and pred >= 0.5:
                        false_pos[object_label[obj, 1]] += 1
                    elif label == 0 and pred < 0.5:
                        true_neg[object_label[obj, 1]] += 1
                print(true_pos)
                self._export_object_analysis(true_pos, false_neg, false_pos, true_neg, model_output_dir)

        # Log the ratios
        pos_lab = labels[labels == 1]
        neg_lab = labels[labels == 0]

        self.logger.info("%d samples, %d grasps" % (len(labels), len(labels) / levels))
        self.logger.info("%d positive grasps, %d negative grasps" % (len(pos_lab) / levels, len(neg_lab) / levels))
        self.logger.info("Model overall accuracy %.2f" % (100 - results.error_rate))
        self.logger.info("True positive samples: %d" % (len(results.true_positive_indices)))
        self.logger.info("True negative samples: %d" % (len(results.true_negative_indices)))
        self.logger.info(
            "Correct predictions: %d" % (len(results.true_positive_indices) + len(results.true_negative_indices)))
        self.logger.info("False positive samples: %d" % (len(results.false_positive_indices)))
        self.logger.info("False negative samples: %d" % (len(results.false_negative_indices)))
        self.logger.info(
            "False predictions: %d" % (len(results.false_positive_indices) + len(results.false_negative_indices)))

        cnt = 0  # Counter for grouping the same images with different noise/depth levels
        if self.num_images is None or self.num_images > len(width_arr):
            self.num_images = len(width_arr)
        for j in range(0, self.num_images):
            try:
                if file_arr[j][1] != file_arr[j - 1][1]:
                    cnt = 0
                else:
                    cnt += 1
            except:
                cnt += 1
            if noise_analysis:
                image = self._plot_grasp(image_arr[j], width_arr[j], results, j, noise_arr=noise_arr)
            elif depth_analysis:
                image = self._plot_grasp(image_arr[j], width_arr[j], results, j, depth_arr=depth_arr)
            elif perturb_analysis or single_analysis:
                print("Plot grasp")
                image = self._plot_grasp(image_arr[j], width_arr[j], results, j, perturb_arr=perturb_arr)
            else:
                image = self._plot_grasp(image_arr[j], width_arr[j], results, j)
            try:
                if noise_analysis or depth_analysis or perturb_analysis or single_analysis:
                    image.save(os.path.join(model_output_dir,
                                            "%05d_%03d_example_%03d.png" % (file_arr[j][0], file_arr[j][1], cnt)))
                else:
                    image.save(os.path.join(model_output_dir, "%05d_%03d.png" % (file_arr[j][0], file_arr[j][1])))
                    data = self.scale(image_arr[j][:, :, 0])
                    image = Image.fromarray(data).convert('RGB').resize((300, 300), resample=Image.NEAREST)
                    image.save(os.path.join(model_output_dir, "%05d_%03d_orig.png" % (file_arr[j][0], file_arr[j][1])))
            except:
                image.save(os.path.join(model_output_dir, "Example_%03d.png" % (cnt)))
        if single_analysis:
            print("Plotting depth image")
            j = int(len(image_arr) / 2)
            # Plot pure depth image without prediction labeling.
            image = self._plot_grasp(image_arr[j], width_arr[j], results, j, plt_results=False)
            image.save(os.path.join(model_output_dir, "Depth_image.png"))
        return results

    def _read_data(self, data_dir, noise=False, depth=False, perturb=False, mixture=False):
        """ Read in the data from the given directory.
        Appends all .npz file into the same array.
        Warning: This might be unsuitable for too many images!
        If the dataset is too big, think about adjusting this to
        predicting on bunch of images at a time."""

        read_file_arr = True
        read_obj_arr = True
        if 'Cornell' in data_dir:
            self.dset = 'Cornell'
        elif 'dexnet' in data_dir or 'DexNet' in data_dir or 'Dexnet' in data_dir:
            self.dset = 'DexNet'
        else:
            self.dset = 'Unknown'

        files = os.listdir(data_dir)
        pose_arr = np.empty([])
        width_arr = np.empty([])
        noise_arr = np.empty([])
        depth_arr = np.empty([])

        labels = []
        numbers = [string[-9:-4] for string in files if '.npz' in string]
        counter = len(list(set(numbers)))
        filenumber = ("{0:05d}").format(0)
        # Read in first file

        image_arr = np.load(data_dir + "depth_ims_tf_table_" + filenumber + ".npz")['arr_0']
        poses = np.load(data_dir + "hand_poses_" + filenumber + ".npz")['arr_0']
        metric_arr = np.load(data_dir + "robust_ferrari_canny_" + filenumber + ".npz")['arr_0']

        if mixture:
            identity_arr = np.load(data_dir + "identifier_" + filenumber + ".npz")['arr_0']

        pose_arr = poses[:, 2:3]
        width_arr = poses[:, -1]
        try:
            file_arr = np.load(data_dir + "files_" + filenumber + ".npz")['arr_0']
        except:
            print("Could not load files_00000.npz")
            print("Will ignore files_XXXXX.npz files")
            read_file_arr = False
        try:
            obj_arr = np.load(data_dir + "object_labels_" + filenumber + ".npz")['arr_0']
        except:
            print("Could not load object labels")
            read_obj_arr = False
        label = 1 * (metric_arr > self.metric_thresh)
        labels = label.astype(np.uint8)
        for i in range(1, counter):
            # Loop through the rest of the files
            filenumber = ("{0:05d}").format(i)
            try:
                image_arr = np.concatenate(
                    (image_arr, np.load(data_dir + "depth_ims_tf_table_" + filenumber + ".npz")['arr_0']))
                poses = np.load(data_dir + "hand_poses_" + filenumber + ".npz")['arr_0']
                metrics = np.load(data_dir + "robust_ferrari_canny_" + filenumber + ".npz")['arr_0']
                if read_file_arr:
                    file_arr = np.concatenate((file_arr, np.load(data_dir + "files_" + filenumber + ".npz")['arr_0']))
            except:
                print("Could not open file with ", filenumber)
                print("Continue.")
                continue
            metric_arr = np.concatenate((metric_arr, metrics))
            pose_arr = np.concatenate((pose_arr, poses[:, 2:3]))
            width_arr = np.concatenate((width_arr, poses[:, -1]))
            label = 1 * (metrics > self.metric_thresh)
            labels = np.append(labels, label.astype(np.uint8))
            if read_obj_arr:
                obj_arr = np.concatenate((obj_arr, np.load(data_dir + "object_labels_" + filenumber + ".npz")['arr_0']))
            if mixture:
                identity_arr = np.concatenate(
                    (identity_arr, np.load(data_dir + "identifier_" + filenumber + ".npz")['arr_0']))
        if noise:
            # Add the noise levels, if analysing noise
            noise_arr = np.load(data_dir + "noise_and_tilting_00000.npz")['arr_0']
            for i in range(1, counter):
                filenumber = "{0:05d}".format(i)
                try:
                    noise_arr = np.concatenate(
                        (noise_arr, np.load(data_dir + "noise_and_tilting_" + filenumber + ".npz")['arr_0']))
                except:
                    print("Could not open noise file with filenumber: ", filenumber)
                    print("Continue")
                    continue
            return image_arr, pose_arr, labels, width_arr, file_arr, noise_arr
        if depth:
            # Add the depth levels, if analysing depth
            depth_arr = np.load(data_dir + "depth_info_00000.npz")['arr_0']
            for i in range(1, counter):
                filenumber = "{0:05d}".format(i)
                try:
                    depth_arr = np.concatenate(
                        (depth_arr, np.load(data_dir + "depth_info_" + filenumber + ".npz")['arr_0']))
                except:
                    print("Could not open depth file with filenumber", filenumber)
                    print("Continue.")
                    continue
            # print("Shape pose_arr: ",pose_arr.shape)
            # print("Shape image_arr: ",image_arr.shape)
            return image_arr, pose_arr, labels, width_arr, file_arr, depth_arr
        if perturb:
            # Add the perturb levels, if analysing perturb
            perturb_arr = np.load(data_dir + "grasp_perturbations_00000.npz")['arr_0']
            for i in range(1, counter):
                filenumber = "{0:05d}".format(i)
                try:
                    perturb_arr = np.concatenate(
                        (perturb_arr, np.load(data_dir + "grasp_perturbations_" + filenumber + ".npz")['arr_0']))
                except:
                    print("Could not open perturb file with filenumber", filenumber)
                    print("Continue.")
                    continue
            # print("Shape pose_arr: ",pose_arr.shape)
            # print("Shape image_arr: ",image_arr.shape)
            return image_arr, pose_arr, labels, width_arr, file_arr, perturb_arr
        if read_obj_arr:
            if mixture:
                return image_arr, pose_arr, labels, width_arr, file_arr, obj_arr, identity_arr

            return image_arr, pose_arr, labels, width_arr, file_arr, obj_arr
        else:
            if read_file_arr:
                return image_arr, pose_arr, labels, width_arr, file_arr, None
            else:
                return image_arr, pose_arr, labels, width_arr, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a GQCNN with Tensorflow on single data")
    parser.add_argument("model_name",
                        type=str,
                        default=None,
                        help="name of model to analyse")
    parser.add_argument("data_dir",
                        type=str,
                        default=None,
                        help="path to where the data is stored")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="path to save the analysis")
    parser.add_argument("--analysis",
                        type=str,
                        default=None,
                        help="Should there be a special analysis? Can be depth or noise.")
    parser.add_argument("--visualisation",
                        type=bool,
                        default=False,
                        help="Visualise the t-SNE embeddings.")

    args = parser.parse_args()
    model_name = args.model_name
    output_dir = args.output_dir
    data_dir = args.data_dir
    analysis_type = args.analysis
    visu = args.visualisation

    # Create model dir.
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../models")
    model_dir = os.path.join(model_dir, model_name)

    if output_dir is None:
        output_dir = EXPER_PATH + "Analysis"
    else:
        output_dir = EXPER_PATH + output_dir

    data_dir = DATA_PATH + data_dir

    # Set model type for diagrams
    if 'Cornell' in model_name:
        model = 'Cornell'
    else:
        model = 'DexNet'

    if 'Cornell' in data_dir:
        dset = 'Cornell'
    else:
        dset = 'DexNet'

    # Set the noise, depth and perturbation analysis
    noise_analysis = False
    depth_analysis = False
    perturb_analysis = False
    single_analysis = False

    if analysis_type == 'noise' or analysis_type == 'Noise':
        noise_analysis = True
    elif analysis_type == 'depth' or analysis_type == 'Depth':
        depth_analysis = True
    elif analysis_type == 'perturbation' or analysis_type == 'Perturbation':
        perturb_analysis = True
    elif analysis_type == 'single' or analysis_type == 'Single':
        single_analysis = True

    # Make the output dir.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Initialise analyser and run analysis.
    analyser = GQCNN_Analyse(model=model, dset=dset)
    if visu:
        analyser.visualise(model_dir, output_dir, data_dir)
    else:
        analyser.run_analysis(model_dir, output_dir, data_dir,
                              noise_analysis, depth_analysis, perturb_analysis, single_analysis)
