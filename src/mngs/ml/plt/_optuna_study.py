#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-13 14:13:54 (ywatanabe)"


def optuna_study(lpath, value_str):
    """
    Loads an Optuna study and generates various visualizations for each target metric.

    Parameters:
    - lpath (str): Path to the Optuna study database.
    - value_str (str): The name of the column to be used as the optimization target.

    Returns:
    - None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mngs
    import optuna
    import pandas as pd

    plt, CC = mngs.plt.configure_mpl(plt, fig_scale=3)

    study = optuna.load_study(study_name=None, storage=lpath)

    sdir = lpath.replace("sqlite:///", "./").replace(".db", "/")

    # To get the best trial:
    best_trial = study.best_trial
    print(f"Best trial number: {best_trial.number}")
    print(f"Best trial value: {best_trial.value}")
    print(f"Best trial parameters: {best_trial.params}")
    print(f"Best trial user attributes: {best_trial.user_attrs}")

    # Merge the user attributes into the study history DataFrame
    study_history = (
        study.trials_dataframe()
        .sort_values(["value"])
        .rename(columns={"value": value_str})
    )

    # Add user attributes to the study history DataFrame
    attrs_df = []
    for trial in study.trials:
        user_attrs = trial.user_attrs
        user_attrs = {k: v for k, v in user_attrs.items()}
        attrs_df.append({"number": trial.number, **user_attrs})
    attrs_df = pd.DataFrame(attrs_df).set_index("number")

    # Updates study history
    study_history = study_history.merge(
        attrs_df, left_index=True, right_index=True, how="left"
    ).set_index("number")
    # study_history = mngs.gen.mv_col(study_history, "bACC", 1)
    mngs.io.save(study_history, sdir + "study_history.csv")
    print(study_history)

    # To visualize the optimization history:
    fig = optuna.visualization.plot_optimization_history(
        study, target_name=value_str
    )
    mngs.io.save(fig, sdir + "optimization_history.png")
    mngs.io.save(fig, sdir + "optimization_history.html")
    plt.close()

    # To visualize the parameter importances:
    fig = optuna.visualization.plot_param_importances(
        study, target_name=value_str
    )
    mngs.io.save(fig, sdir + "param_importances.png")
    mngs.io.save(fig, sdir + "param_importances.html")
    plt.close()

    # To visualize the slice of the study:
    fig = optuna.visualization.plot_slice(study, target_name=value_str)
    mngs.io.save(fig, sdir + "slice.png")
    mngs.io.save(fig, sdir + "slice.html")
    plt.close()

    # To visualize the contour plot of the study:
    fig = optuna.visualization.plot_contour(study, target_name=value_str)
    mngs.io.save(fig, sdir + "contour.png")
    mngs.io.save(fig, sdir + "contour.html")
    plt.close()

    # To visualize the parallel coordinate plot of the study:
    fig = optuna.visualization.plot_parallel_coordinate(
        study, target_name=value_str
    )
    mngs.io.save(fig, sdir + "parallel_coordinate.png")
    mngs.io.save(fig, sdir + "parallel_coordinate.html")
    plt.close()


if __name__ == "__main__":
    mngs.plt.configure_mpl(plt, fig_scale=3)
    lpath = "sqlite:///scripts/ml/clf/sub_conv_transformer/optuna_studies/optuna_study_v032.db"
    optuna_study(lpath, "Validation loss")
    # scripts/ml/clf/sub_conv_transformer/optuna_studies/optuna_study_v032
