import json
import os
import matplotlib.pyplot as plt


# CONFIG
DATASET_NAME = "de-en"   # change when needed


def load_losses(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["train"], data["val"]


base_path = f"results/{DATASET_NAME}"

# Load losses
sin_train, sin_val = load_losses(
    f"{base_path}/sinusoidal/losses.json"
)

learned_train, learned_val = load_losses(
    f"{base_path}/learned/losses.json"
)

rot_train, rot_val = load_losses(
    f"{base_path}/rotary/losses.json"
)

epochs = range(1, len(sin_train) + 1)

# Plot
plt.figure(figsize=(8, 6))

# Training losses (dashed)
plt.plot(epochs, sin_train, linestyle="--", marker="o",
         label="Sinusoidal (Train)")
plt.plot(epochs, learned_train, linestyle="--", marker="o",
         label="Learned (Train)")
plt.plot(epochs, rot_train, linestyle="--", marker="o",
         label="Rotary (Train)")

# Validation losses (solid)
plt.plot(epochs, sin_val, marker="s",
         label="Sinusoidal (Val)")
plt.plot(epochs, learned_val, marker="s",
         label="Learned (Val)")
plt.plot(epochs, rot_val, marker="s",
         label="Rotary (Val)")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training & Validation Loss Comparison ({DATASET_NAME})")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save globally
os.makedirs("results/plots", exist_ok=True)

plt.savefig(
    f"results/plots/{DATASET_NAME}_loss_comparison.png",
    dpi=300
)

plt.show()
