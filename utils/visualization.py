import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation(image, output, label_colors, save_path=None):
    image = image.squeeze(0).permute(1, 2, 0).numpy()
    output = output.squeeze(0).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow((image * 255).astype(np.uint8))
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    seg_image = label_colors[output]
    ax[1].imshow(seg_image)
    ax[1].set_title('Segmentation Output')
    ax[1].axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()