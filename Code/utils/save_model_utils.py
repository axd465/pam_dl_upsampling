# Customary Imports:
import os
from utils.history_utils import save_history
##################################################################################################################################
'''
MODEL HISTORY FUNCTIONS:
'''
def save_model(unet_model, output_dir, model_dir_list = ['model_dir','model_dir_ssim_psnr'], latest_model_filename = 'saved_model.h5'):
    
    if not os.path.exists(os.path.join(os.getcwd(), output_dir)):
        print('Creating Directory...')
        os.mkdir(os.path.join(os.getcwd(), output_dir))
    else:
        print('Directory Already Exists...')

    #'''
    print('Saving Latest Model...')
    directory = os.path.join(os.getcwd(), output_dir)
    directory = os.path.join(directory, latest_model_filename)
    unet_model.save(directory)
    print('Saving History...')
    save_history(history, output_dir)
    #'''
    for model_dir in model_dir_list:
        best_model_dir = os.path.join(output_dir, f'best_models_from_{model_dir}')
        if not os.path.exists(best_model_dir):
            print(f'Creating ./{model_dir} Directory...')
            os.mkdir(best_model_dir)
        else:
            print(f'./{model_dir} Directory Already Exists...')
        print(f'Saving Best Model From ./{model_dir}...')

        filename = sorted(os.listdir(model_dir), key = lambda x : int(x.partition('h_')[2].partition('-S')[0]))[-1]
        directory = os.path.join(os.getcwd(), model_dir)
        directory = os.path.join(directory, filename)
        os.rename(directory, os.path.join(best_model_dir, filename))
        filename = sorted(os.listdir(model_dir), key = lambda x : int(x.partition('h_')[2].partition('-S')[0]))[-1]
        directory = os.path.join(os.getcwd(), model_dir)
        directory = os.path.join(directory, filename)
        os.rename(directory, os.path.join(best_model_dir, filename))
    print('Done Saving !!!')