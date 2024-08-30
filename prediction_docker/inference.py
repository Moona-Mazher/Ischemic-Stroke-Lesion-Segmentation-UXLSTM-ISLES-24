# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:55:10 2024

@author: aq22
"""

import torch
import os
import numpy as np
import SimpleITK as sitk
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import time
def run():
    _show_torch_cuda_info()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    start_time = time.time()
    nnUNet_results = "/opt/app/resources/nnUNet_results"
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda',0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset119_SELS-MRG/nnUNetTrainerUxLSTMBot__nnUNetPlans__3d_fullres'),
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )
    
    input_folder = "/input/images/preprocessed-tmax-map/"
    output_folder = "/output/images/stroke-lesion-segmentation/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    input_files = glob(str(input_folder + "/*.tiff")) + glob(str(input_folder + "/*.mha"))
    input_file_name = input_files[0]
    output_file_name = output_folder + "/output.mha"

    # predict a single numpy array
    img, props = SimpleITKIO().read_images([input_file_name])
    print("img.shape: ", img.shape)
    print("props: ", props)
    pred_array = predictor.predict_single_npy_array(img, props, None, None, False)
    pred_array = pred_array.astype(np.uint8)
    print("pred_array.shape: ", pred_array.shape)
    print("pred_array_labels_before: ", np.unique(pred_array))
    # # Reverse Label mapping
    # reverse_label_map = {
    #     19: 21, 20: 22, 21: 23, 22: 24,
    #     23: 25, 24: 26, 25: 27, 26: 28,
    #     27: 31, 28: 32, 29: 33, 30: 34,
    #     31: 35, 32: 36, 33: 37, 34: 38,
    #     35: 41, 36: 42, 37: 43, 38: 44,
    #     39: 45, 40: 46, 41: 47, 42: 48
    # }
    
    # # Apply the reverse mapping
    # pred_array = np.vectorize(lambda x: reverse_label_map.get(x, x))(pred_array)
    # print("pred_array_labels after: ", np.unique(pred_array))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for prediction: {elapsed_time:.4f} seconds")
    torch.cuda.empty_cache()

    image = sitk.GetImageFromArray(pred_array)
    image.SetDirection(props['sitk_stuff']['direction'])
    image.SetOrigin(props['sitk_stuff']['origin'])
    image.SetSpacing(props['sitk_stuff']['spacing'])
    #image = sitk.Cast(image, sitk.sitkUInt8)
    sitk.WriteImage(
        image,
        output_file_name,
        useCompression=True,
    )
                                 
    print('Saved!!!')
    return 0

def _show_torch_cuda_info():

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())