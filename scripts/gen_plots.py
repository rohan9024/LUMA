# scripts/gen_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results():
    """Generates and saves plots for the paper."""
    sns.set_theme(style="whitegrid")

    # --- Data from your runs ---
    baseline_data = {
        'Metric': ['Recall@1', 'Recall@5', 'Recall@10', 'MRR'],
        'Score': [0.3871, 0.8065, 0.9355, 0.5859]
    }

    memory_data = {
        'Metric': ['Group Recall@10', 'Group MRR'],
        'Score': [0.5371, 0.4750]
    }

    audio_data = {
        'Metric': ['Recall@10', 'MRR'],
        'Score': [0.4194, 0.1082]
    }
    
    # --- Plot 1: Baseline Performance ---
    df_base = pd.DataFrame(baseline_data)
    plt.figure(figsize=(8, 5))
    bar1 = sns.barplot(x='Metric', y='Score', data=df_base)
    plt.title('Text-to-Image Retrieval Performance (Baseline)', fontsize=16)
    plt.ylabel('Score')
    plt.xlabel('')
    plt.ylim(0, 1.0)
    for p in bar1.patches:
        bar1.annotate(format(p.get_height(), '.2f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')
    plt.tight_layout()
    plt.savefig('paper_plot_baseline.png', dpi=300)
    print("Saved paper_plot_baseline.png")

    # --- Plot 2: Memory Offload Performance ---
    df_mem = pd.DataFrame(memory_data)
    plt.figure(figsize=(8, 5))
    bar2 = sns.barplot(x='Metric', y='Score', data=df_mem)
    plt.title('Retrieval with Memory Offloading (Group-Aware)', fontsize=16)
    plt.ylabel('Score')
    plt.xlabel('')
    plt.ylim(0, 1.0)
    for p in bar2.patches:
        bar2.annotate(format(p.get_height(), '.2f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')
    plt.tight_layout()
    plt.savefig('paper_plot_memory.png', dpi=300)
    print("Saved paper_plot_memory.png")

    # --- Plot 3: Audio Performance ---
    df_audio = pd.DataFrame(audio_data)
    plt.figure(figsize=(8, 5))
    bar3 = sns.barplot(x='Metric', y='Score', data=df_audio)
    plt.title('Audio-to-Image Retrieval Performance', fontsize=16)
    plt.ylabel('Score')
    plt.xlabel('')
    plt.ylim(0, 1.0)
    for p in bar3.patches:
        bar3.annotate(format(p.get_height(), '.2f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')
    plt.tight_layout()
    plt.savefig('paper_plot_audio.png', dpi=300)
    print("Saved paper_plot_audio.png")


if __name__ == '__main__':
    plot_results()