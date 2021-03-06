import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from path_settings import EXPER_PATH

"""
Script to analyse the absolute prediction errors of a trained (and previously analysed)
GQCNN model. Reads in the training and validation labels/predictions of the previously
analysed model (what has been outputted by gqcnn/analyze_gqcnn_performance.py).
Creates .png files for the absolute prediction error on training/validation data of
positive and negative samples.
The plots can be configured for a white or a black background for slides.
"""


class Absolute_Prediction_Error():
    def __init__(self):
        self.black_background = False  # Set to false if your slides have a white background

    def create_errors(self, data_dir, output_dir):
        # Read in the data
        train_labels, train_predictions, val_labels, val_predictions = read_data(data_dir)
        all_labels = np.concatenate([train_labels, val_labels])
        all_predictions = np.concatenate([train_predictions, val_predictions])

        # Calculate the prediction errors
        self.train_error_pos, self.train_error_neg = calculate_prediction_errors(train_predictions, train_labels)
        self.val_error_pos, self.val_error_neg = calculate_prediction_errors(val_predictions, val_labels)
        self.all_error_pos, self.all_error_neg = calculate_prediction_errors(all_predictions, all_labels)

        # Plot and save the histograms
        self._calculate_accuracy()
        self._plot_errors(output_dir)

        return None

    def _calculate_accuracy(self):
        # Calculate validation accuracy, all accuracy and accuracy on successful and unsuccessful grasps
        # Validation

        val_correct = len(np.where(self.val_error_pos <= 0.5)[0]) + len(np.where(self.val_error_neg <= 0.5)[0])
        val_images = len(self.val_error_pos) + len(self.val_error_neg)
        print("Validation accuracy is: ", val_correct / val_images * 100, "%")

        all_correct = len(np.where(self.all_error_pos <= 0.5)[0]) + len(np.where(self.all_error_neg <= 0.5)[0])
        all_images = len(self.all_error_pos) + len(self.all_error_neg)
        print("Accuracy is: ", all_correct / all_images * 100, "%")

        pos_correct = len(np.where(self.all_error_pos <= 0.5)[0])
        pos_images = len(self.all_error_pos)
        print("Accuracy on successful images is: ", pos_correct / pos_images * 100, "%")

        neg_correct = len(np.where(self.all_error_neg <= 0.5)[0])
        neg_images = len(self.all_error_neg)
        print("Accuracy on unsuccessful images is: ", neg_correct / neg_images * 100, "%")

        val_pos_correct = len(np.where(self.val_error_pos <= 0.5)[0])
        val_pos_images = len(self.val_error_pos)
        print("Accuracy on successful validation images is: ", val_pos_correct / val_pos_images * 100, "%")

        val_neg_correct = len(np.where(self.val_error_neg <= 0.5)[0])
        val_neg_images = len(self.val_error_neg)
        print("Accuracy on unsuccessful validation images is: ", val_neg_correct / val_neg_images * 100, "%")

        return None

    def _plot_errors(self, output_dir):
        # Plot the absolute prediction errors. Adjust the line and histogram
        # color accordingly to the background color
        binwidth = 0.02
        if self.black_background:
            line_color = 'w'
            hist_color = (0.616, 0.773, 0.730)
        else:
            line_color = 'k'
            hist_color = (0.2, 0.6, 1)
        plt.rc('axes', edgecolor=line_color)
        plt.rc('text', color=line_color)
        plt.rc(('xtick', 'ytick'), c=line_color)

        plt.hist(self.train_error_pos, bins=np.arange(0, 1 + binwidth, binwidth), color=hist_color)
        plt.xlabel("Absolute Prediction Error", color=line_color, fontsize=14)
        plt.title("Error on positive training grasps", fontsize=18)
        plt.axvline(x=0.5, color='red')
        plt.savefig(output_dir + "err_pos_train", transparent=True)
        plt.close()

        plt.hist(self.train_error_neg, bins=np.arange(0, 1 + binwidth, binwidth), color=hist_color)
        plt.xlabel("Absolute Prediction Error", color=line_color, fontsize=14)
        plt.title("Error on negative training grasps", fontsize=18)
        plt.axvline(x=0.5, color='red')
        plt.savefig(output_dir + "err_neg_train", transparent=True)
        plt.close()

        plt.hist(self.val_error_pos, bins=np.arange(0, 1 + binwidth, binwidth), color=hist_color)
        plt.xlabel("Absolute Prediction Error", color=line_color, fontsize=14)
        plt.title("Error on positive validation grasps", fontsize=18)
        plt.axvline(x=0.5, color='red')
        plt.savefig(output_dir + "err_pos_val", transparent=True)
        plt.close()

        plt.hist(self.val_error_neg, bins=np.arange(0, 1 + binwidth, binwidth), color=hist_color)
        plt.xlabel("Absolute Prediction Error", color=line_color, fontsize=14)
        plt.title("Error on negative validation grasps", fontsize=18)
        plt.axvline(x=0.5, color='red')
        plt.savefig(output_dir + "err_neg_val", transparent=True)
        plt.close()

        plt.hist(self.all_error_pos, bins=np.arange(0, 1 + binwidth, binwidth), color=hist_color)
        plt.xlabel("Absolute Prediction Error", color=line_color, fontsize=14)
        plt.title("Error on positive grasps", fontsize=18)
        plt.axvline(x=0.5, color='red')
        plt.savefig(output_dir + "err_pos_all", transparent=True)
        plt.close()

        plt.hist(self.all_error_neg, bins=np.arange(0, 1 + binwidth, binwidth), color=hist_color)
        plt.xlabel("Absolute Prediction Error", color=line_color, fontsize=14)
        plt.title("Error on negative grasps", fontsize=18)
        plt.axvline(x=0.5, color='red')
        plt.savefig(output_dir + "err_neg_all", transparent=True)
        plt.close()


def calculate_prediction_errors(predictions, labels):
    pos_ind = np.where(labels == 1)
    neg_ind = np.where(labels == 0)
    pos_prediction_errors = np.abs(labels[pos_ind] - predictions[pos_ind])
    neg_prediction_errors = np.abs(labels[neg_ind] - predictions[neg_ind])

    return pos_prediction_errors, neg_prediction_errors


def read_data(data_dir):
    train_labels = np.load(data_dir + "/train_result.cres/labels.npz")['arr_0']
    train_predictions = np.load(data_dir + "/train_result.cres/predictions.npz")['arr_0']
    val_labels = np.load(data_dir + "/val_result.cres/labels.npz")['arr_0']
    val_predictions = np.load(data_dir + "/val_result.cres/predictions.npz")['arr_0']
    return train_labels, train_predictions, val_labels, val_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create CSV files for the absolute prediction errors")
    parser.add_argument("data_dir",
                        type=str,
                        default=None,
                        help="path to where the data is stored")

    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = EXPER_PATH + 'Abs_pred_error/'

    # Make the output dir.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    saver = Absolute_Prediction_Error()
    saver.create_errors(data_dir, output_dir)
