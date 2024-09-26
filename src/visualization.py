import matplotlib.pyplot as plt
import seaborn as sns

def visualize_true_vs_predicted(y_test, y_pred, y_label):
# Set up the figure for side-by-side plots
    plt.figure(figsize=(14, 6))

    # First subplot: Actual vs Predicted plot
    plt.subplot(1, 2, 1)  # (1 row, 2 columns, first plot)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Line of perfect prediction
    plt.title(f"Actual vs Predicted {y_label}")
    plt.xlabel(f"Actual {y_label}")
    plt.ylabel(f"Predicted {y_label}")

    # Second subplot: Residual plot
    residuals = y_test - y_pred
    plt.subplot(1, 2, 2)  # (1 row, 2 columns, second plot)
    plt.scatter(y_pred, residuals, alpha=0.7, color='g')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel(f"Predicted {y_label}")
    plt.ylabel("Residuals")

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()