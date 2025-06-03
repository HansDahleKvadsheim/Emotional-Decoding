import nibabel as nib
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np
from nilearn import datasets, plotting
from nilearn.image import new_img_like, index_img
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch, Rectangle


class Explainer:
    def __init__(self):
        self.reverse_dictionary = self.create_reverse_dict()

    


    def create_reverse_dict(self):
        atlas_path = "../../Alice/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_3mm.nii"
        atlas_img = nib.load(atlas_path)
        data = atlas_img.get_fdata().astype(int)

        roi_to_voxel = defaultdict(list)

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    roi = data[x, y, z]
                    if (roi > 0):
                        roi_to_voxel[roi].append((x,y,z))

        roi_to_voxels = dict(roi_to_voxel)
        return roi_to_voxels




    # def visualize_explaination(self, rois, scores):
    #     path = "../../Alice/Subjects/sub-18/sub-18_task-alice_bold_preprocessed.nii.gz"
    #     images = nib.load(path).get_fdata()
    #     img = nib.load(path).get_fdata()[:,:,:,1]
    #     atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    #     num_slices = img.shape[2]

    #     cols = 7  
    #     rows = math.ceil(num_slices / cols)  
    #     plt.style.use('default')
        
        
    #     fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    #     colors_dict = {"red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255], "yellow": [255, 255, 0], "cyan": [0, 255, 255]}
    #     colors = ['red', 'green', 'blue', 'yellow', 'cyan']



    #     for i in range(num_slices):
    #         ax = axes[i // cols, i % cols]  
            
    #         grayscale_img = img[:,:, i]
    #         rgb_img = np.repeat(grayscale_img[:,:, np.newaxis], 3, axis=2)
    #         grayscale_img = (grayscale_img - np.min(grayscale_img)) / (np.max(grayscale_img) - np.min(grayscale_img)) * 255  
    #         grayscale_img = grayscale_img.astype(np.uint8)
    #         rgb_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)

    #         for j, roi in enumerate(rois):
    #             voxels = self.reverse_dictionary[roi]
    #             for voxel in voxels:
    #                 if voxel[-1] == i:
    #                     rgb_img[voxel[0], voxel[1], :] = colors_dict[colors[j]]

    #         ax.imshow(rgb_img)
    #         ax.set_title(f"Slice {i+1}") 
    #         ax.axis('off') 

    

    #     for i in range(num_slices, rows * cols):
    #         fig.delaxes(axes.flatten()[i])


    #     plt.subplots_adjust(hspace=0.5)
    #     plt.figtext(0.5, 0.95, "Most predictive ROIs", fontsize=20, ha='center', va='top')
        
        
    #     starting_pos = 0.1

    #     for i, score in enumerate(scores):
    #         plt.figtext(0.5, starting_pos, f"{i+1}: {atlas['labels'][rois[i]].decode('utf-8')}, score = {score}", color=colors[i], fontsize=12, ha='center', va='top')
    #         starting_pos -= 0.02
    #     fig.patch.set_facecolor('lightgray')
    #     plt.show()
    #     plt.tight_layout()


    # def visualize_3d(self, rois, scores):
    #     path = "../../Alice/Subjects/sub-18/sub-18_task-alice_bold_preprocessed.nii.gz"
    #     images = nib.load(path).get_fdata()
    #     data = nib.load(path).get_fdata()[:,:,:,1]
    #     img = np.copy(data)
    #     atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    #     num_slices = img.shape[2]

    #     colors_dict = {"red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255], "yellow": [255, 255, 0], "cyan": [0, 255, 255]}
    #     colors = ['red', 'green', 'blue', 'yellow', 'cyan']

    #     image = []

    #     for i in range(num_slices):
            
    #         grayscale_img = img[:, :, i]
    #         rgb_img = np.repeat(grayscale_img[:,:, np.newaxis], 3, axis=2)
    #         grayscale_img = (grayscale_img - np.min(grayscale_img)) / (np.max(grayscale_img) - np.min(grayscale_img)) * 255  
    #         grayscale_img = grayscale_img.astype(np.uint8)
    #         rgb_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
    #         for j, roi in enumerate(rois):
    #             voxels = self.reverse_dictionary[roi]
    #             for voxel in voxels:
    #                 if voxel[-1] == i:
    #                     rgb_img[voxel[0], voxel[1], :] = colors_dict[colors[j]]

    #         image.append(rgb_img)

    

        


        # return image
            
    def visualize_3d(self, rois, scores, emotion):
        path = "../../Alice/Subjects/sub-18/sub-18_task-alice_bold_preprocessed.nii.gz"
        img_4d = nib.load(path)
        img = index_img(img_4d, 1)
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
        labels = atlas.labels

        data = img.get_fdata()
        modified_data = np.zeros_like(data)

        color_names = ["black", "red", "green", "blue", "yellow", "cyan"]
        cmap = mcolors.ListedColormap(color_names)

        for j, roi in enumerate(rois):
            voxels = self.reverse_dictionary[roi+1]
            for x, y, z in voxels:
                modified_data[x, y, z] = j + 1

        colored_img = new_img_like(img, modified_data)

        fig = plt.figure(figsize=(10, 8))
        grid_spec = fig.add_gridspec(2, 1, height_ratios=[4, 1])

        ax_brain = fig.add_subplot(grid_spec[0])
        display = plotting.plot_glass_brain(
            colored_img,
            cmap=cmap,
            display_mode='lyrz',
            colorbar=False,
            black_bg=True,
            axes=ax_brain,
            title=f"Most Important ROIs for {emotion}"
        )


                
        ax_legend = fig.add_subplot(grid_spec[1])
        ax_legend.axis('off')

        roi_labels = [labels[roi].decode("utf-8").replace("7Networks_", "") for roi in rois]
        line_height = 0.20
        start_y = 1
        center_x = 0.32
        box_width = 0.03
        gap = 0.01  

        for i, (label, score) in enumerate(zip(roi_labels, scores)):
            y = start_y - i * line_height
            color = color_names[i + 1]

            box_x = center_x - (box_width + gap) / 2
            text_x = center_x + (box_width + gap) / 2

            rect = Rectangle((box_x, y - box_width / 2), box_width, box_width,
                            transform=ax_legend.transAxes, facecolor=color, edgecolor='none')
            ax_legend.add_patch(rect)

            ax_legend.text(text_x, y, f"{label}, importance score: {score:.3f}",
                        transform=ax_legend.transAxes,
                        fontsize=12, va='center', ha='left', color='black')
