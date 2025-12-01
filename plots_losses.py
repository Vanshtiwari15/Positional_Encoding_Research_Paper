import json
import matplotlib.pyplot as plt


def load_losses(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["train"], data["val"]


# -------- Load loss logs --------
sin_train, sin_val = load_losses("results/logs/sinusoidal_losses.json")
learnt_train, learnt_val = load_losses("results/logs/learned_losses.json")
rot_train, rot_val = load_losses("results/logs/rotary_losses.json")

epochs = range(1, len(sin_train) + 1)

# -------- Plot --------
plt.figure(figsize=(8, 6))

# Training losses
plt.plot(epochs, sin_train, marker="o", linestyle="--", label="Sinusoidal (Train)")
plt.plot(epochs, learnt_train, marker="o", linestyle="--", label="Learned (Train)")
plt.plot(epochs, rot_train, marker="o", linestyle="--", label="Rotary (Train)")

# Validation losses
plt.plot(epochs, sin_val, marker="s", label="Sinusoidal (Val)")
plt.plot(epochs, learnt_val, marker="s", label="Learned (Val)")
plt.plot(epochs, rot_val, marker="s", label="Rotary (Val)")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss vs Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("results/plots/loss_comparison.png", dpi=300)
plt.show()
