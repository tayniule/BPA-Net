import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def visualize_tsne(self, model_path=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)


    for name, param in checkpoint.items():
        if 'leaner' in name and 'weight' in name:
            layer_name = name.replace('.weight', '')
            input_dim = param.shape[1]
            output_dim = param.shape[0]
            setattr(self.model, layer_name, nn.Linear(input_dim, output_dim).to(device))

    self.model.load_state_dict(checkpoint, strict=False)
    self.model.eval()

    shared_list, private_list = [], []
    subject_labels, task_labels = [], []

    with torch.no_grad():
        for batch in self.test_data_loader:
            eeg, eog, emg, y, l, d = batch
            self.model(to_gpu(eeg), to_gpu(eog), to_gpu(emg), to_gpu(l), to_gpu(d), self.groups)

            shared_list.append(self.model.utt_shared_subject.cpu().numpy())
            private_list.append(self.model.utt_private_subject.cpu().numpy())
            subject_labels.append(d.cpu().numpy())
            task_labels.append(y.cpu().numpy())


    self._plot_tsne_by_group(shared_list, private_list, subject_labels, task_labels)


def _plot_tsne_by_group(self, shared_list, private_list, subject_labels, task_labels):
    all_subs = np.concatenate(subject_labels, axis=0)
    all_tasks = np.concatenate(task_labels, axis=0)

    # --- 核心逻辑：建立 Subject 到 Group 的映射 ---
    # 假设 self.groups 格式为: {group_id: [sub1, sub2, ...]}
    sub_to_group = {}
    for g_id, subs in self.groups.items():
        for s in subs:
            sub_to_group[int(s)] = int(g_id)

    # 将测试集中的每个点映射到其对应的 Group ID
    # 如果某个被试不在分组里，默认设为 -1
    group_color_labels = np.array([sub_to_group.get(int(s), -1) for s in all_subs])

    def draw_plot(data_list, color_data, title, filename, is_task=False):
        data = np.concatenate(data_list, axis=0)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        data_2d = tsne.fit_transform(data)
        data_2d = MinMaxScaler().fit_transform(data_2d)

        plt.figure(figsize=(12, 11))

        # 使用 tab10 配色，因为 group_num 通常较小 (2, 3, 5 等)
        cmap = 'coolwarm' if is_task else 'tab10'
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=color_data,
                              cmap=cmap, s=80, alpha=0.7, edgecolors='none')

        plt.title(title, fontsize=42, fontweight='bold', pad=30)
        plt.xticks([]);
        plt.yticks([])

        cbar = plt.colorbar(scatter)
        # 动态设置颜色条标签
        if is_task:
            cbar.set_ticks([0, 1]);
            cbar.set_ticklabels(['Low', 'High'])
        else:
            # 显示 Group 0, Group 1...
            unique_groups = np.unique(color_data)
            cbar.set_ticks(unique_groups)
            cbar.set_ticklabels([f'Group {int(g)}' for g in unique_groups])

        cbar.ax.tick_params(labelsize=26)

        os.makedirs('tsne_results', exist_ok=True)
        plt.savefig(f'tsne_results/{filename}.png', dpi=300)
        plt.close()


    # Shared 特征：按 Group 染色（预期：不同 Group 的点混在一起，证明去除了组间差异）
    draw_plot(shared_list, group_color_labels, "Shared Features (by Group)", "shared_by_group")

    # Private 特征：按 Group 染色（预期：形成 group_num 个明显的簇）
    draw_plot(private_list, group_color_labels, "Private Features (by Group)", "private_by_group")
