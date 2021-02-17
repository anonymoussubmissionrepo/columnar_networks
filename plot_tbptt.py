import logging
import os

logger = logging.getLogger('experiment')

import itertools
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import experiment as exp
from utils import utils
import configs.plots.plot as reg_parser
import ast

results_dict = {}
std_dict = {}
all_experiments = []
folders = []
all_experiments = {}
legend = []
colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k'))


def main():
    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])

    args = utils.get_run(all_args, rank)

    my_experiment = exp.experiment(args["name"], args, args["output_dir"], commit_changes=False,
                                   rank=int(rank / total_seeds),
                                   seed=total_seeds)

    my_experiment.results["all_args"] = all_args

    param_dict = {}
    results_dir = args["path"]
    for experiment in os.listdir(results_dir):
        if "DS_St" not in experiment:
            # print(experiment)
            for run in os.listdir(os.path.join(results_dir, experiment)):
                if "DS_St" not in run:

                    print(os.path.join(results_dir, experiment, run, "metadata.json"))
                    try:
                        path = os.path.join(results_dir, experiment, run, "metadata.json")
                        with open(path) as json_file:

                            data_temp = json.load(json_file)
                            experiment_name = str(data_temp['params'])
                            param_dict[experiment_name] = data_temp['params']
                            if experiment_name in all_experiments:
                                all_experiments[experiment_name].append(data_temp['results'][args["metric"]])
                                # print(data_temp['results']['Real_Error_list'])
                            else:

                                all_experiments[experiment_name] = [data_temp['results'][args["metric"]]]
                                # print(data_temp['results']['Real_Error_list'])

                    except:

                        pass

    sns.set(style="whitegrid")
    sns.set_context("paper", font_scale=0.4, rc={"lines.linewidth": 1.0})



    truncation_dict = []
    for experiment_name in all_experiments:
        experiment_params = param_dict[experiment_name]
        temp = experiment_params["truncation"]
        if temp not in truncation_dict:

            truncation_dict.append(temp)

    for a in truncation_dict:
        counter = 0
        x = []
        y = []
        error = []
        for experiment_name in all_experiments:

            experiment_params = param_dict[experiment_name]

            if experiment_params["width"] == 50 and experiment_params["columns"] == 20 and experiment_params["truncation"] == a:

                for list_of_vals in all_experiments[experiment_name]:
                    # print(list_of_vals)
                    # print(d.strip("[").strip("]").strip("\,"))
                    y_pred = ast.literal_eval(list_of_vals)
                    y_pred = [float(x) for x in y_pred]

                    y_pred_mean = np.mean(y_pred)
                    y_pred_error = np.std(y_pred) / np.sqrt(len(y_pred))
                    x_cur = experiment_params["sparsity"]
                    x.append(x_cur)
                    y.append(y_pred_mean)
                    # print(x)
                    error.append(y_pred_error)
                    # d_sparse = []
                    # running_sum = y_pred[0]
                    # for number, value_in in enumerate(y_pred):
                    #     running_sum = running_sum * 0.96 + value_in * 0.04
                    #     if number % 50 == 0:
                    #         d_sparse.append(running_sum)
                    # print(d_new)

        x = np.array(x)
        if args["log"]:
            x = np.log10(x)
        y = np.array(y)
        arg_sort = np.argsort(x)
        x = np.array([x[p] for p in arg_sort])
        y = np.array([y[p] for p in arg_sort])
        error = np.array([error[p] for p in arg_sort])

        plt.fill_between(x, y - error, y + error, alpha=0.4)
        plt.plot(x, y)
        plt.ylim(0.4, 1)
        # plt.xlim(0, 10)


        plt.tight_layout()
        print(my_experiment.path + "result.pdf")
    plt.legend(truncation_dict)
    plt.savefig(my_experiment.path + "result.pdf", format="pdf")


if __name__ == "__main__":
    main()