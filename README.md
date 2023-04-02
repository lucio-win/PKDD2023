# GPRGNN-L

This is the source code for our ECML-PKDD 2023 paper:

# Repreduce results in our paper:
We save all the results of our experiments in the results folder, the naming for the file is:

acc_10_{Init}_{dataset}_{train_rate}_{model}.npy.

gamma_10_{Init}_{dataset}_{train_rate}_{model}.npy.

gamma_10_Random_{dataset}_unsupervised.

where Init can be chosen from Random (GPRGNN-R) and WS (GPRGNN-L) and model can be chosen from gprgnn and bernnet.

acc_* record the accuracy of ten runs and gamma_*_{model}.npy record the learned filter.

gamma_10_Random_{dataset}_unsupervised is the filter learned in the LP task.

''plot_filter.ipynb'' can be used to display the filters, while ''performance_under_diffsplits_all.ipynb'' can be used to display the accuracy of different approaches.

To generate the results from scratch, you can run the ''link_pred_hetero_gprgnn.ipynb'' in the script folder to learn an optimal filter first. After that, you can run the ''train_model_fix_filter_hetero_gprgnn.py'' to evaluate both GPRGNN-R and GPRGNN-L under different splits.

