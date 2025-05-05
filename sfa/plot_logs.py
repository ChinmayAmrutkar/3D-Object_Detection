from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

log_path = "../logs/fpn_resnet_18/tensorboard"  # Update this if you moved it

ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

print("Available tags:", ea.Tags()["scalars"])

for tag in ea.Tags()["scalars"]:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    plt.plot(steps, values, label=tag)

plt.xlabel("Steps")
plt.ylabel("Value")
plt.title("Training Metrics")
plt.legend()
plt.grid()
plt.show()
