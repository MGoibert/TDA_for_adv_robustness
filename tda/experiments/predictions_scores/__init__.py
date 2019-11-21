from r3d3 import ExperimentDB
import matplotlib.pyplot as plt


def plot_scatter(db: ExperimentDB, run_id: int, epsilon: str):

    df = db.show_experiment(
        run_id,
        params={},
        metrics={
            "pred_clean": f"predictions.{epsilon}.pred_clean",
            "pred_noisy": f"predictions.{epsilon}.pred_noisy",
            "pred_adv": f"predictions.{epsilon}.pred_adv",
        })

    data = dict(df.iloc[0])

    fig = plt.figure()
    fig.set_figheight(12)

    for i, pair in enumerate([
        ("pred_clean", "pred_adv"),
        ("pred_clean", "pred_noisy"),
        ("pred_noisy", "pred_adv")
    ]):
        ax = fig.add_subplot(310 + i + 1)
        ax.plot(data[pair[0]], data[pair[0]], 'r-')
        ax.scatter(data[pair[0]], data[pair[1]])
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])