import os
from viz import DropoutVisualizer


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "viz")
    os.makedirs(output_dir, exist_ok=True)

    model_layers = [5, 8, 6, 4, 3]
    dropout_rates = [0.0, 0.2, 0.4, 0.6]
    total_steps = 40

    visualizer = DropoutVisualizer(
        layers=model_layers,
        dropout_rates=dropout_rates,
        total_steps=total_steps,
        output_dir=output_dir,
        filename="dropout_training.mp4",
    )

    visualizer.run()
    print(f"Video saved to {os.path.join(output_dir, 'dropout_training.mp4')}")


if __name__ == "__main__":
    main()
