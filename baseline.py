# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TASK 1

# %%
import cv2
from matplotlib import pyplot as plt
import sys
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime


# %%
def image_loader(img_path):
    tmp_img_path_list = os.listdir(img_path)
    tmp_img_path_list.sort()
    start_img = tmp_img_path_list[-1]
    del tmp_img_path_list[-1]
    tmp_img_path_list.insert(0, start_img) # To make the start.jpg to be the first one in the array
    
    img_array_list = []
    for img in tmp_img_path_list:
        tmp_path = os.path.join(img_path, img)
        img_arr = cv2.imread( tmp_path, cv2.IMREAD_GRAYSCALE)
        img_arr_origin = cv2.imread(tmp_path)
        img_array_list.append((img_arr, img_arr_origin))
    return img_array_list


# %%
def weight_calc(inverse, size):
    if inverse==True:
        k = 2
    else :
        k = -2
    # Calculation
    ls_to_cvt = []
    for x in range(0, size): # x would be u or n
        for a in range(0, size): # a would be m or v
            tmp = np.exp((k * np.pi * (0 + 1.j) * a * x) / size)
            ls_to_cvt.append(tmp)
    arr_to_return = np.array(ls_to_cvt)
    arr_to_return = np.reshape(arr_to_return, (-1, size))
    
    # if inverse == True:
    #     arr_to_return = np.transpose(arr_to_return) 
    return arr_to_return

def weight_M(M, inverse):
    if inverse==False:
        weight_M = weight_calc(False, M)
    else :
        weight_M = weight_calc(True, M)
    return weight_M

def weight_N(N, inverse):
    if inverse==False:
        weight_N = weight_calc(False, N)
    else :
        weight_N = weight_calc(True, N)
    return weight_N

def fourier_Transform(img, patch_size, w_M, w_N, sRow, sCol, inverse): # img for image input, patch_size for patch input, sRow for starting pixel in row level, sCol for starting pixel in col level, inverwse tfor whether you are up to inverse transform or not
    M, N = patch_size
    img = img[sRow : sRow + M, sCol : sCol + N]
    if inverse==False:
        return_map = (1 / (M * N)) * ((w_M @ img) @ w_N)
    else:
        return_map = (w_M @ img) @ w_N
    return return_map

# %% [markdown]
# # TASK 2

# %%
def phase_correlation(ref_map, to_compare, patch_size, w_M_inverse, w_N_inverse):
    before_inverse = np.multiply(ref_map, np.conj(to_compare)) / np.abs(np.multiply(ref_map, np.conj(to_compare)))
    M, N = patch_size
    after_inverse = fourier_Transform(before_inverse, patch_size, w_M_inverse, w_N_inverse, 0, 0, True)
    # phase_val = after_inverse[int(M/2), int(N/2)].real # SHOULD RECHECK FOR THE RETURN VALUE OF THE np.angle
    # # print(phase_var)
    # phase_val = np.sum(np.abs(after_inverse[int(M/2) - 2 : int(M/2 + 3), int(N/2) - 2 : int(N/2) + 3]))
    # phase_val = np.abs(np.sum(after_inverse[0:2, 0:2]) + np.sum(after_inverse[int(M) - 2: int(M), int(N) - 2: int(N)]) + np.sum(after_inverse[0:2, int(N) - 2: int(N)]) + np.sum(after_inverse[int(M) - 2 : int(M), 0:2]))
    # phase_val = np.abs(np.sum(after_inverse[int(M/2) - 1: int(M/2) + 1, int(N/2) - 1: int(N/2) + 1]))
    # phase_val = np.sum(np.abs(after_inverse))
    # phase_val = np.max(np.angle(after_inverse))
    phase_val = np.max(after_inverse.real)
    return phase_val


# %%
# def non_maximum_suppression(img):


# %%
def main():
    img_path = '/home/carpedkm/GitHub/Signals_and_Systems/dataset (1)/'
    save_dir = '/home/carpedkm/GitHub/Signals_and_Systems/result_activation'
    patch_size = (26, 30)
    M, N = patch_size
    img_list = image_loader(img_path)
    print('[IMAGE LOAD COMPLETE]')
    w_M = weight_M(M, False)
    w_N = weight_N(N, False)

    w_M_inverse = weight_M(M, True)
    w_N_inverse = weight_N(N, True)
    # print(img_list[0][137:167, 270:300])
    # plt.imshow(img_list[0][138:162, 276:294])
    # plt.show()
    reference_patch_ftmap = fourier_Transform(img_list[0][0], patch_size, w_M, w_N, sRow=137, sCol=275, inverse=False)
    # img = img_list[0][0][137:163, 275:304]
    # reference_patch_ftmap = np.fft.fft2(img)
    inverse_of_ref_ftmap = fourier_Transform(reference_patch_ftmap, patch_size, w_M_inverse, w_N_inverse, sRow=0, sCol=0, inverse=True)
    # print('[PHASE CORREL CALC start]')
    # temp_patch = img_list[0][0:M, 0:N]
    # corel_val = phase_correlation(temp_patch, reference_patch_ftmap, patch_size, w_M_inverse, w_N_inverse)
    # print(corel_val[1, 1])
    # print(corel_val[1, 1])
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    save_dir = save_dir + '_' + str(patch_size[0]) + '_' + str(patch_size[1]) + '_' + date_time
    os.makedirs(save_dir, exist_ok=True)

     # Reference patch imshow
    temp_patch_rf = np.abs(reference_patch_ftmap)
    temp_patch_rf = (temp_patch_rf  - np.min(temp_patch_rf)) / (np.max(temp_patch_rf) - np.min(temp_patch_rf))
    temp_patch_rf[0, 0] = 0
    plt.imshow(temp_patch_rf)
    plt.show()
    # return
    # return

    for cnt, (img, origin) in enumerate(img_list):
        # kernel = np.ones((5, 5), np.float32)/ 25
        # img = cv2.filter2D(img, -1, kernel)
        # img = cv2.equalizeHist(img)
        print('IMAGE', cnt, 'START')
        act_tmp = []
        max_corel_val = 0
        r_temp = 0
        c_temp = 0
        padding = (int(M/2), int(N/2))
        # mean_of_img = np.sum(img) / (img.shape[0] * img.shape[1])
        # img = np.pad(img, ((padding[0], padding[0]), (padding[1], padding[1]), 'constant', constant_values=mean_of_img)
        # img[img > 110] = 0
        # img[img < 80] = 0
        plt.imshow(img)
        plt.show()
        for i in tqdm(range(0, 360 - M, 1)):
            for j in range(0, 480 - N, 1):
                temp_patch = img[i:i+M, j:j+N]
                temp_patch = fourier_Transform(temp_patch, patch_size, w_M, w_N, sRow=0, sCol=0, inverse=False)
                # temp_patch = np.fft.fft2(temp_patch)
                corel_val = phase_correlation(temp_patch, reference_patch_ftmap, patch_size, w_M_inverse, w_N_inverse)
                
                if max_corel_val < corel_val:
                    max_corel_val = corel_val
                    r_temp = i
                    c_temp = j
                act_tmp.append(corel_val)

        tmp = cv2.rectangle(origin, (c_temp, r_temp), (c_temp + N, r_temp + M), (255, 255, 255), 3)
        plt.imshow(tmp)
        plt.show()
        # ACTIVATION MAP
        act_tmp = np.array(act_tmp)
        activation_map_tmp = 'activation_' + str(cnt) + '.jpg'
        annotated_temp = 'result_' + str(cnt) + '.jpg'
        

        to_save_activation = os.path.join(save_dir, activation_map_tmp)
        to_save_annotation = os.path.join(save_dir, annotated_temp)
        act_tmp = np.reshape(act_tmp, (-1, 480 - N))
       
        # ACTIVATION TO SAVE : normalizing
        abs_patch = act_tmp
        patch_activation = (abs_patch - np.min(abs_patch)) / (np.max(abs_patch) - np.min(abs_patch))
        patch_activation = (255 * patch_activation).astype(int)

        plt.imshow(patch_activation)
        plt.show()
             
        # act_tmp = cv2.normalize(abs_patch, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imwrite(to_save_activation, patch_activation)

         # ANNOTATION TO SAVE
        cv2.imwrite(to_save_annotation, tmp)

        # if cnt == 0:
        #     return


# %%
main()


# %%
#  if i == 2 and j == 2 :
#                     to_show = np.abs(temp_patch)
#                     to_show = (to_show - np.min(to_show)) / (np.max(to_show) - np.min(to_show))
#                     to_show[0, 0] = 0
#                     plt.imshow(to_show)
#                     plt.show()
#                     tmp_img = img[i:i+M, j:j+N]
#                     to_show_img = (tmp_img - np.min(tmp_img)) / (np.max(tmp_img) - np.min(tmp_img))
#                     plt.imshow(to_show_img)
#                     plt.show()
#                 if i == 136 and j == 273 :
#                     to_show = np.abs(temp_patch)
#                     to_show = (to_show - np.min(to_show)) / (np.max(to_show) - np.min(to_show))
#                     to_show[0, 0] = 0
#                     plt.imshow(to_show)
#                     plt.show()
#                     tmp_img = img[i:i+M, j:j+N]
#                     to_show_img = (tmp_img - np.min(tmp_img)) / (np.max(tmp_img) - np.min(tmp_img))
#                     plt.imshow(to_show_img)
#                     plt.show()
#                 if i == 359 - M and j == 479 - N:
#                     to_show = np.abs(temp_patch)
#                     to_show = (to_show - np.min(to_show)) / (np.max(to_show) - np.min(to_show))
#                     to_show[0, 0] = 0
#                     plt.imshow(to_show)
#                     plt.show()
#                     tmp_img = img[i:i+M, j:j+N]
#                     to_show_img = (tmp_img - np.min(tmp_img)) / (np.max(tmp_img) - np.min(tmp_img))
#                     plt.imshow(to_show_img)
#                     plt.show()


# %%



