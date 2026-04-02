import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 本地弹窗显示图片

feature_map = np.load('/data/lzn/INFAF/runs/vis/fea/stage43_C3_features.npy')

print(feature_map.shape)  # 打印形状


if feature_map.ndim == 3:
    for i in range(min(5, feature_map.shape[0])):
        plt.imshow(feature_map[i], cmap='viridis')
        plt.colorbar()
        plt.title(f"Feature Map - Channel {i}")
        plt.savefig(f"feature_channel{i}.png", dpi=300, bbox_inches='tight')
        plt.close()



# elif feature_map.ndim == 2:
#     plt.imshow(feature_map, cmap='viridis')
#     plt.colorbar()
#     plt.title("Feature Map")
#     plt.savefig("feature_map.png", dpi=300, bbox_inches='tight')  # 保存图像
#     plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
#
# matplotlib.use('TkAgg')
#
# for name in ["attfusion", "ghostnet", "mobilenet"]:
#     # feature_map = np.load(f'./{name}/12-features.npy')
#     feature_map = np.load(f'./{name}/13-features.npy')
#
#     print("Feature map shape:", feature_map.shape)
#
#
#     if feature_map.ndim == 3:
#         num_channels_to_show = min(3, feature_map.shape[0])
#
#
#         fig, axes = plt.subplots(1, num_channels_to_show, figsize=(5 * num_channels_to_show, 4))
#
#         if num_channels_to_show == 1:
#             axes = [axes]
#
#         for i in range(num_channels_to_show):
#             im = axes[i].imshow(feature_map[i], cmap='viridis')
#             axes[i].set_title(f"Feature Map - Channel {i}")
#             axes[i].axis('off')
#             plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
#
#         plt.tight_layout()
#         # plt.savefig("feature_maps_combined.png", dpi=300, bbox_inches='tight')
#         plt.show()
