# Customary Imports:
import os
import re
from skimage import exposure
import shutil
import imageio
from pathlib import Path

##################################################################################################################################
'''
CHANGES DATA DOWNSAMPLING RATIO WHILE STILL MAINTAINING ORIGINAL DATA SPLIT:
'''
def extract_full_samp_and_retain_file_struct(input_dir, output_dir, delete_previous = True):
    '''
    This function loops through an input directory and converts each file to undo the
    function "pad_img_and_add_down_channel." The modified image is then saved into the 
    output directory.
    '''
    print('Starting Extraction...')
    abs_input_dir = os.path.join(os.getcwd(), input_dir)
    folder_list = os.listdir(abs_input_dir)
    abs_output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.exists(abs_output_dir):
        os.mkdir(abs_output_dir)
    elif delete_previous == True:
        shutil.rmtree(abs_output_dir)
        os.mkdir(abs_output_dir)
    for folder in folder_list:
        if folder != '.ipynb_checkpoints':
            folder_dir = os.path.join(abs_input_dir, folder)
            folder_dir = os.path.join(folder_dir, 'input')
            file_list = os.listdir(folder_dir)
            new_dir = os.path.join(abs_output_dir, folder)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
                new_dir = os.path.join(new_dir, 'input')
                os.mkdir(new_dir)
            for file in file_list:
                filename = os.fsdecode(file)
                filepath = os.path.join(folder_dir, filename)
                if filepath.endswith('.npy'):
                    array = np.load(filepath)
                else:
                    array = imageio.imread(filepath)
                if len(array.shape) == 3:
                    array = array[..., 0] # First Index Represents Fully Sampled Image
                else: 
                    print('Error: please only input data directory that has already been formatted using function \"standardize_dir\"')
                if filename[-5].isdigit() and '-' in filename[-9:]:
                    file_ext = filename[-4:]
                    regex = re.compile(r'_[^_]+$')
                    filename = re.sub(regex, "", filename) + file_ext
                new_filepath = os.path.join(new_dir, filename)
                if filepath.endswith('.npy'):
                    new_filepath = Path(new_filepath)
                    np.save(new_filepath, array, allow_pickle=True, fix_imports=True)
                else:
                    new_filepath = Path(new_filepath)
                    imageio.imwrite(new_filepath, array)
    print('Ending Extraction !!!')