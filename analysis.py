import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# تنظیمات اولیه برای ظاهر زیباتر نمودارها
sns.set_theme(style="whitegrid")


def analyze_results(csv_file="experimental_results.csv"):
    """
    داده‌های نهایی نتایج را خوانده و نمودارهای کلیدی را برای گزارش تولید می‌کند.
    """
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' was not found.")
        print("Please run 'project.py' first to generate the final results.")
        return

    # خواندن داده‌های نهایی از فایل CSV
    df = pd.read_csv(csv_file)
    print("Successfully loaded final results. Data summary:")
    print(df)

    # --- نمودار ۱: مقایسه میانگین وظایف تکمیل‌شده ---
    plt.figure(figsize=(10, 7))
    sns.barplot(data=df, x="config_name", y="avg_tasks_completed", hue="agent_type", palette="viridis")
    plt.title("Comparison of Average Tasks Completed", fontsize=16, fontweight='bold')
    plt.xlabel("Experimental Scenario", fontsize=12)
    plt.ylabel("Average Tasks Completed (out of 2)", fontsize=12)
    plt.ylim(0, 2.5)  # محدود کردن محور Y برای نمایش بهتر
    plt.legend(title="Agent Type")

    output_path_tasks = "final_comparison_tasks.png"
    plt.savefig(output_path_tasks)
    print(f"\n✓ Final chart saved to {output_path_tasks}")

    # --- نمودار ۲: مقایسه میانگین زمان تکمیل وظیفه (مهم‌ترین نمودار) ---
    plt.figure(figsize=(10, 7))
    # فقط داده‌هایی را رسم می‌کنیم که وظیفه‌ای در آن‌ها انجام شده باشد
    plot_data = df[df['avg_completion_time'] > 0]

    sns.barplot(data=plot_data, x="config_name", y="avg_completion_time", hue="agent_type", palette="plasma")
    plt.title("Comparison of Average Task Completion Time (Efficiency)", fontsize=16, fontweight='bold')
    plt.xlabel("Experimental Scenario", fontsize=12)
    plt.ylabel("Average Steps to Complete All Tasks", fontsize=12)
    plt.legend(title="Agent Type")

    output_path_efficiency = "final_comparison_efficiency.png"
    plt.savefig(output_path_efficiency)
    print(f"✓ Final Efficieny chart saved to {output_path_efficiency}")

    plt.show()


if __name__ == "__main__":
    analyze_results()