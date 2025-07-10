import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# تنظیمات اولیه برای ظاهر زیباتر نمودارها
sns.set_theme(style="whitegrid")


def analyze_results(csv_file="experimental_results.csv"):
    """
    داده‌های نتایج را از فایل CSV خوانده و نمودارهای مقایسه‌ای تولید می‌کند.
    """
    # بررسی وجود فایل نتایج
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' was not found.")
        print("Please run 'project.py' first to generate the results.")
        return

    # خواندن داده‌ها با استفاده از کتابخانه pandas
    # نام ستون‌ها را به ترتیب صحیح به صورت دستی تعریف می‌کنیم
    column_names = ["config_name", "agent_type", "tasks_completed", "total_steps", "average_energy_remaining"]
    df = pd.read_csv(csv_file, header=None, names=column_names)
    print("Successfully loaded results. Data summary:")
    print(df)

    # --- نمودار ۱: مقایسه تعداد وظایف تکمیل‌شده ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="config_name", y="tasks_completed", hue="agent_type")
    plt.title("Comparison of Tasks Completed by Agent Type", fontsize=16)
    plt.xlabel("Experimental Scenario", fontsize=12)
    plt.ylabel("Number of Tasks Completed", fontsize=12)
    plt.xticks(rotation=10)  # چرخش لیبل‌های محور x برای خوانایی بهتر
    plt.legend(title="Agent Type")

    # ذخیره نمودار در فایل
    output_path_tasks = "comparison_tasks_completed.png"
    plt.savefig(output_path_tasks)
    print(f"\n✓ Chart saved to {output_path_tasks}")

    # --- نمودار ۲: مقایسه تعداد گام‌های طی شده (بهره‌وری) ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="config_name", y="total_steps", hue="agent_type")
    plt.title("Comparison of Steps Taken (Efficiency)", fontsize=16)
    plt.xlabel("Experimental Scenario", fontsize=12)
    plt.ylabel("Total Steps Taken", fontsize=12)
    plt.xticks(rotation=10)
    plt.legend(title="Agent Type")

    # ذخیره نمودار در فایل
    output_path_steps = "comparison_total_steps.png"
    plt.savefig(output_path_steps)
    print(f"✓ Chart saved to {output_path_steps}")

    # --- نمودار ۳: مقایسه انرژی باقی‌مانده ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="config_name", y="average_energy_remaining", hue="agent_type")
    plt.title("Comparison of Average Energy Remaining", fontsize=16)
    plt.xlabel("Experimental Scenario", fontsize=12)
    plt.ylabel("Average Energy Remaining", fontsize=12)
    plt.xticks(rotation=10)
    plt.legend(title="Agent Type")

    # ذخیره نمودار در فایل
    output_path_energy = "comparison_energy_remaining.png"
    plt.savefig(output_path_energy)
    print(f"✓ Chart saved to {output_path_energy}")

    plt.show()  # نمایش نمودارها پس از ذخیره


if __name__ == "__main__":
    analyze_results()