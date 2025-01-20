import matplotlib.pyplot as plt
import pandas as pd



def plot_loss(df_logs: pd.DataFrame):
    df_train_step_loss = df_logs[["step", "train_loss"]][~df_logs["train_loss"].isna()]
    df_train_epoch_loss = df_logs.groupby(["epoch"]).agg({"step": "max", "train_loss": "mean"})[["step", "train_loss"]]
    df_val_epoch_loss = df_logs[["step", "val_loss"]][~df_logs["val_loss"].isna()]
    last_step_per_epoch = df_train_epoch_loss["step"]
    
    plt.plot(df_train_epoch_loss["step"], df_train_epoch_loss["train_loss"], label="Train loss (epoch)")
    plt.plot(df_val_epoch_loss["step"], df_val_epoch_loss["val_loss"], label="Val loss (epoch)")
    plt.plot(df_train_step_loss["step"], df_train_step_loss["train_loss"], "--", alpha=0.6, label="Train loss (step)")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    sec = plt.gca().secondary_xaxis(location="top")
    sec.set_xticks(last_step_per_epoch, labels=range(len(last_step_per_epoch)))
    sec.set_xlabel("Epoch")
    plt.ylim(top=df_train_epoch_loss["train_loss"].max() * 1.1) # Ignore first steps of train loss as it is (usually) too large
    plt.legend()
    plt.grid()


def plot_miou(df_logs: pd.DataFrame):
    df_train_epoch_miou = df_logs[["epoch", "train_miou"]][~df_logs["train_miou"].isna()]
    df_val_epoch_miou = df_logs[["epoch", "val_miou"]][~df_logs["val_miou"].isna()]

    plt.plot(df_train_epoch_miou["epoch"], df_train_epoch_miou["train_miou"], label="Train mIoU (epoch)")
    plt.plot(df_val_epoch_miou["epoch"], df_val_epoch_miou["val_miou"], label="Val mIoU (epoch)")

    plt.ylabel("mIoU")
    plt.xlabel("Epoch")
    plt.xticks(df_train_epoch_miou["epoch"])
    plt.legend()
    plt.grid()


def plot_logs(logs_path: str):
    df_logs = pd.read_csv(logs_path)

    plt.figure(figsize=(15, 3))
    plt.subplot(1, 2, 1)
    plot_loss(df_logs)
    plt.subplot(1, 2, 2)
    plot_miou(df_logs)
    
    plt.show()