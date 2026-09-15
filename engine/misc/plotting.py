"""Per-epoch train/val loss & accuracy curve plotting (matplotlib, Agg backend)."""

import os


class MetricsPlotter:
    """Accumulates per-epoch scalars and rewrites a loss/accuracy PNG after each update.

    Call ``update()`` once per epoch from the main process only.
    """

    def __init__(self, output_dir: str, filename: str = "training_curves.png", acc_label: str = "Accuracy"):
        self.output_dir = output_dir
        self.filename = filename
        self.acc_label = acc_label
        self.epochs: list[int] = []
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.train_acc: list[float] = []
        self.val_acc: list[float] = []

    def update(self, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self._plot()

    def _plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[MetricsPlotter][WARN] matplotlib unavailable: {e}")
            return

        try:
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

            ax_loss.plot(self.epochs, self.train_loss, label="train", marker=".")
            ax_loss.plot(self.epochs, self.val_loss, label="val", marker=".")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("Loss")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)

            ax_acc.plot(self.epochs, self.train_acc, label="train", marker=".")
            ax_acc.plot(self.epochs, self.val_acc, label="val", marker=".")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel(self.acc_label)
            ax_acc.set_title(self.acc_label)
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)

            fig.tight_layout()
            os.makedirs(self.output_dir, exist_ok=True)
            fig.savefig(os.path.join(self.output_dir, self.filename))
            plt.close(fig)
        except Exception as e:
            print(f"[MetricsPlotter][WARN] failed to save {self.filename}: {e}")
