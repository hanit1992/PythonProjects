# we will implement two main functions: a function who finds a liver segmentation, and a function who measures the
# success, comparing to some ground truth. but first, we will implement small functions which will help us with the
# implementation of those 2 main functions
import random

import nibabel as nib
import numpy as np
from skimage import measure,morphology
import math
from collections import deque

# note: in this excersice i assume that the positive direction of the coronal axis is the front (meanning that as closer
#  to the front of the body i am, the bigger is the axis coronal slice)

BLACK = 0
WHITE = 1
MIN_BODY_TH = -500
MAX_BODY_TH = 2000
BODY_IMAGE_NOISE_FILTER_TH = 200
NUM_OF_SEEDS = 200
ADDITION_TO_GET_LIVER_ROI = 30
SUBSTRACTION_OF_MAX_SAGITAL_SEED_ROI = 30
ADDITION_TO_MIN_SAGITAL_SEED_ROI = 150
SUBSTRACTION_OF_MAX_CORONAL_SEED_ROI = 100
#TODO need to play with those 2 to get a full roi contains the liver in all situations, as a bbox to the region growing search
MAX_LIVER_CORONAL_SUBSTRACTION = 10
MIN_LIVER_CORONAL_ADDTION = -40
AXIAL_SLICE_ADDITION_TO_FIND_SEEDS = 2
AXIAL_SLICE_ADDITION = 30
MIN_LIVER_SAGITAL_ADDITION = 120
MAX_LIVER_SAGITAL_SUBSTRUCTION = 20

"""
this function will take a threshold of some image data(pointer to a nifty file) - and will change it 
"""
def take_threshold(min, max, scan_data):
    in_range_pixels = np.logical_and(scan_data <= max, scan_data >= min)
    out_range_pixels = np.logical_or(scan_data > max, scan_data < min)
    scan_data[in_range_pixels] = WHITE
    scan_data[out_range_pixels] = BLACK

"""
this function will get the size of the biggest component of some labeled array
"""
def get_size_of_biggest_component(array_of_labeled_connectivity):
    # lets get props of regions first, so we can get the sizes of the components easily
    region_props = measure.regionprops(array_of_labeled_connectivity)
    max_area_size = 0
    for region in region_props:
        if region.area > max_area_size:
            max_area_size = region.area
    return max_area_size


def get_nbhd(curr, checked, shape):
    x, y, z = curr
    relevant_neighbors = []
    relevant_neighbors = [(x + i, y + j, z + k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)
                     if (x+i>0 and  x+i < shape[0]-1 and y+j >0 and y+j < shape[1]-1 and z+k>0 and z+k<shape[2]-1
                     and not checked[x+i,y+j,z+k])]
    print(888)


    # # relevant_neighbors.remove((x, y, z))
    # relevant_neighbors = [location for location in relevant_neighbors]

    return relevant_neighbors



    # nbhd = []
    #
    # if (curr[0] > 0) and not checked[curr[0] - 1, curr[1], curr[2]]:
    #     nbhd.append((curr[0] - 1, curr[1], curr[2]))
    # if (curr[1] > 0) and not checked[curr[0], curr[1] - 1, curr[2]]:
    #     nbhd.append((curr[0], curr[1] - 1, curr[2]))
    # if (curr[2] > 0) and not checked[curr[0], curr[1], curr[2] - 1]:
    #     nbhd.append((curr[0], curr[1], curr[2] - 1))
    #
    # if (curr[0] < shape[0] - 1) and not checked[curr[0] + 1, curr[1], curr[2]]:
    #     nbhd.append((curr[0] + 1, curr[1], curr[2]))
    # if (curr[1] < shape[1] - 1) and not checked[curr[0], curr[1] + 1, curr[2]]:
    #     nbhd.append((curr[0], curr[1] + 1, curr[2]))
    # if (curr[2] < shape[2] - 1) and not checked[curr[0], curr[1], curr[2] + 1]:
    #     nbhd.append((curr[0], curr[1], curr[2] + 1))
    #
    # return nbhd


def alternative_region_growing(ct_scan,ROI):
    seeds_roi = get_liver_roi_for_seeds_search(ROI)
    # full_roi_img = get_ROI_image_from_binary_ROI(ct_scan,ROI)
    # first, we want to get the seeds. we will get them randomly from the liver ROI we found
    seeds_of_region = find_seeds(ct_scan, seeds_roi)
    # lets get the ct image data
    ct_scan_data = ct_scan.get_data()
    liver_seg = np.zeros(ct_scan_data.shape)
    checked = np.zeros_like(liver_seg)
    # indexes = tuple(np.array(list(zip(*seeds_of_region))))
    liver_seg[seeds_of_region[0]] = True
    checked[seeds_of_region[0]] = True
    checked[True] = np.where(ROI == BLACK, True, checked)
    need_to_check = get_nbhd(seeds_of_region[0],checked,ct_scan_data.shape)

    while len(need_to_check)>0:
        print("in region: "+ str(np.sum(liver_seg)))
        # if(np.sum(liver_seg)>20000):
        #     break
        print("to check: "+str(len(need_to_check)))
        curr = need_to_check.pop()
        if checked[curr]:
            continue
        checked[curr] = True

        imin = max(curr[0] - 5, 0)
        imax = min(curr[0] + 5, ct_scan_data.shape[0] - 1)
        jmin = max(curr[1] - 5, 0)
        jmax = min(curr[1] + 5, ct_scan_data.shape[1] - 1)
        kmin = max(curr[2] - 5, 0)
        kmax = min(curr[2] + 5, ct_scan_data.shape[2] - 1)

        if ct_scan_data[curr] >= ct_scan_data[imin:imax + 1, jmin:jmax + 1, kmin:kmax + 1].mean():
            liver_seg[curr] = True
            need_to_check += get_nbhd(curr, checked, ct_scan_data.shape)

    return liver_seg



"""
this function will clean image from small connectivity elements- and then will take the biggest one
"""
def clean_image_and_get_biggest_connectivity(image_data_to_clean):
    # first lets activate the label function in order to get an array of connectivity components, each is labeled with
    # some int
    array_of_labeled_connectivity, num_of_components = measure.label(image_data_to_clean,return_num=True)
    # first lets get rid of small connectivity components
    image_data_to_clean[True] = morphology.remove_small_objects(array_of_labeled_connectivity,BODY_IMAGE_NOISE_FILTER_TH)
    morphology.binary_opening(image_data_to_clean,out=image_data_to_clean)
    # now lets get the size of the biggest component and only return it
    array_of_labeled_connectivity = measure.label(image_data_to_clean)
    image_data_to_clean[True] = morphology.remove_small_objects(array_of_labeled_connectivity,
                                                                get_size_of_biggest_component(array_of_labeled_connectivity)-1)
    # #TODO - just to check if it's one component
    # print(len(measure.regionprops(array_of_labeled_connectivity)))

"""
this function will create a new nifty image name according to a given addition
"""
def get_name_to_save_nifty_file(original_name,addition):
    name_with_no_postfix = original_name.strip("nii.gz")
    return name_with_no_postfix+addition+".nii.gz"

"""
this function will take a CT scan and will return the segmentation of the whole body. it will create the segmentation
by taking a threshold, then cleaning and getting the largest connectivity component
input: a CT scan - nifty object - of the body
output: the nifty object after we made the segmentation
"""
def IsolateBody(CT_scan):
    # first lets get the scan DATA
    scan_data = CT_scan.get_data()

    # now, lets take a threshold - the threshold is based on the knowledge we have on the intensity range values of the
    # whole body
    take_threshold(MIN_BODY_TH,MAX_BODY_TH,scan_data)

    # now lets filter the noise and get a good segmentation - by getting rid of small components and taking the biggest
    # component
    clean_image_and_get_biggest_connectivity(scan_data)
    #print(measure.label(scan_data,return_num=True)[1])

    return CT_scan

def get_ROI_image_from_binary_ROI(CT_scan, ROI):
    # lets get the data of the scan
    scan_data = CT_scan.get_data()

    # now lets create the ROI image
    image_ROI = np.where(ROI == WHITE, scan_data, 0)

    return image_ROI

def get_liver_roi_for_seeds_search(full_roi):
    components_array = measure.label(full_roi)
    props = measure.regionprops(components_array)
    min_axial = props[0].bbox[2]
    max_axial = props[0].bbox[5]
    min_coronal = props[0].bbox[1]
    max_coronal = props[0].bbox[4]
    min_sagital = props[0].bbox[0]
    max_sagital = props[0].bbox[3]
    center = (min_axial+max_axial)/2
    new_min_axial = center-AXIAL_SLICE_ADDITION_TO_FIND_SEEDS
    new_max_axial = center+AXIAL_SLICE_ADDITION_TO_FIND_SEEDS
    new_min_sagital = min_sagital+ADDITION_TO_MIN_SAGITAL_SEED_ROI
    new_max_sagital = max_sagital-SUBSTRACTION_OF_MAX_SAGITAL_SEED_ROI
    new_max_coronal = max_coronal - SUBSTRACTION_OF_MAX_CORONAL_SEED_ROI
    mask = np.zeros(full_roi.shape)
    mask[int(new_min_sagital):int(new_max_sagital),:int(new_max_coronal),int(new_min_axial):int(new_max_axial)] = True
    new_roi = np.where(mask==True,full_roi,0)
    return new_roi

"""
this function will find seeds from a given ROI, it will randomly choose an N number of seeds - meanning, points tuples,
and return a list of them 
"""
def find_seeds(CT_scan,ROI):

    # lets get the image only in the ROI that was given to us
    ct_scan_ROI = get_ROI_image_from_binary_ROI(CT_scan,ROI)

    # and lets get the props of the region
    label_connectivity_array = measure.label(ROI)
    props = measure.regionprops(label_connectivity_array)

    # now we want to get all N seeds from the ROI
    all_seeds = [(seed[0],seed[1],seed[2]) for seed in props[0].coords if ct_scan_ROI[(seed[0],seed[1],seed[2])]<200
                 and ct_scan_ROI[(seed[0],seed[1],seed[2])]>-100]
    #choose randomly from all ROI coords, N points
    len_of_arr = len(all_seeds)
    random_indexes = random.sample(range(len_of_arr),NUM_OF_SEEDS)

    seeds_final = [all_seeds[i] for i in random_indexes]

    return seeds_final

"""
this method will expand the pixel - will get all it's 3d neighbors 
"""
def get_26_neighbors(pixel):
    x,y,z = pixel
    all_neighbors = [(x + i, y + j, z + k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
    all_neighbors.remove((x, y, z))
    all_neighbors = [location for location in all_neighbors]
    return all_neighbors

def check_if_pixel_is_in_bbox_for_region_growing(bbox,pixel):
    min_sagital,min_coronal,min_axial,max_sagital,max_coronal,max_axial = bbox
    if(pixel[0]>=min_sagital and pixel[0]<=max_sagital and pixel[1]>= min_coronal and pixel[1]<=max_coronal
       and pixel[2]>=min_axial and pixel[2]<=max_axial):
        return True
    else:
        return False

def check_if_pixel_in_region(pixel,ct_scan_data,current_region_mean,labeled_pixels,pixels_to_investigate,bbox):
    if(ct_scan_data[pixel] != 0 and
       ct_scan_data[pixel] <= current_region_mean + 2 and
       ct_scan_data[pixel] >= current_region_mean - 2 and
       labeled_pixels[pixel]==False and
       pixel not in pixels_to_investigate and
       ct_scan_data[pixel] <= 200 and
       ct_scan_data[pixel] >= -100 and
       check_if_pixel_is_in_bbox_for_region_growing(bbox, pixel)):
        return True
    return False

def sort_by_intensity_func(pixel,ct_scan_data):
    return ct_scan_data[pixel]

# def unmark_all_the_out_of_bounds(array,bbox,x,y,z):
#     min_sagital, min_coronal, min_axial, max_sagital, max_coronal, max_axial = bbox
#     if (x-1<min_sagital):


"""
this function will return the segmentation based on a randomly chose seeds from the given ROI, by using the region
growing segmentation method
"""
def multipleSeedsRegionGrowing(ct_scan, ROI):

    seeds_roi = get_liver_roi_for_seeds_search(ROI)
    # first, we want to get the seeds. we will get them randomly from the liver ROI we found
    seeds_of_region = find_seeds(ct_scan,seeds_roi)
    # lets get the ct image data
    ct_scan_data = ct_scan.get_data()
    # we are only searching in the big ROI
    ct_scan_data = np.where(ROI==WHITE,ct_scan_data,0)

    # now, when we have the seeds, we will start the algorithm.
    # we will have 3 lists, in all of them the initial elements will be the seeds - a list of all the pixels in the
    # region, a list of all the pixels to check their neighbors, and a list of all the pixels we already labeled

    labeled_pixels = np.zeros(shape=ct_scan_data.shape)
    out_seg_img = np.zeros(shape=ct_scan_data.shape)

    indexes = tuple(np.array(list(zip(*seeds_of_region))))
    labeled_pixels[indexes] = 1
    labeled_pixels[True] = np.where(ROI == BLACK, 1, labeled_pixels)
    out_seg_img[indexes] = 1

    current_region_intensities = ct_scan_data[np.where(out_seg_img == True)]
    current_region_mean = np.mean(current_region_intensities)
    pixels_to_investigate = set(seeds_of_region)

    i = 0
    inside_range_neighbors_mask = np.zeros(ct_scan_data.shape)
    # while not (np.all(labeled_pixels)):
    while len(pixels_to_investigate)>0:
        #TODO add a break condition that if all is labeled - to break
        i=i+1

        # if i==300:
            # break
        # breaking when all is labeled
        if (np.all(labeled_pixels)):
            break

        curr_pixel = pixels_to_investigate.pop()

        x,y,z = curr_pixel

        # lets take off the out of bonus pixels

        cond = (ct_scan_data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]!=0)&\
               (ct_scan_data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]<=current_region_mean+20) &\
               (ct_scan_data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]>=current_region_mean-20) &\
               (labeled_pixels[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]==False)
        # we will not include the pixels outside the roi
        inside_range_neighbors_mask[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = np.where(ROI[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]==WHITE,cond,0)

        # 3. add to investigate
        addition = [(i, j, k) for i in range(x - 1, x + 2) for j in range(y - 1, y + 2) for k in range(z - 1, z + 2)
                    if (inside_range_neighbors_mask[(i, j, k)] == True and labeled_pixels[(i, j, k)] == False
                    and out_seg_img[(i, j, k)] == False and not np.all(out_seg_img[i-1:i+2,j-1:j+2,k-1:k+2])
                    and ct_scan_data[(i,j,k)]!=0)]

        # 1. fill in the seg
        out_seg_img[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = np.where(inside_range_neighbors_mask[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2],1,
                                                                      out_seg_img[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2])
        # 2. update the labeled
        pixels_to_investigate.update(addition)
        labeled_pixels[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = True
        # setting all array back to 0
        inside_range_neighbors_mask[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = False

    return out_seg_img

def get_min_coronal_bounding_from_aorta_seg(aorta_seg_data,center_axial):
    aorta_seg_data = aorta_seg_data[:,:,int(center_axial)]
    aorta_slice_label_array = measure.label(aorta_seg_data)
    props = measure.regionprops(aorta_slice_label_array)
    max_coronal = props[0].bbox[3]
    liver_min_coronal = max_coronal + MIN_LIVER_CORONAL_ADDTION
    return liver_min_coronal

def get_max_coronal_bounding_from_body_seg(body_label_array):
    props = measure.regionprops(body_label_array)
    max_coronal = props[0].bbox[4]
    liver_max_coronal = max_coronal - MAX_LIVER_CORONAL_SUBSTRACTION
    return liver_max_coronal

def get_axial_bounding_from_aorta_seg(aorta_label_array):
    props = measure.regionprops(aorta_label_array)
    min_axial_slice = props[0].bbox[2]
    max_axial_slice = props[0].bbox[5]
    center_axial = (min_axial_slice+max_axial_slice)/2

    return center_axial-AXIAL_SLICE_ADDITION,center_axial+AXIAL_SLICE_ADDITION,center_axial

def get_sagital_bounderies_from_body_seg_slices(body_segmentation,min_axial,max_axial):
    body_seg_in_slices = body_segmentation[:,:,int(min_axial):int(max_axial)]
    body_label_array = measure.label(body_seg_in_slices)
    props = measure.regionprops(body_label_array)
    min_sagital = props[0].bbox[0]+MIN_LIVER_SAGITAL_ADDITION
    max_sagital = props[0].bbox[3]-MAX_LIVER_SAGITAL_SUBSTRUCTION
    return min_sagital,max_sagital

"""
this function will get a ROI which contained in the liver area. it will use an aorta place in order to get the liver ROI,
and will use the body segmentation as well. basically - we will focus only on the central Aorta slices and get the roi 
from them. 
returns: a liver ROI to get the seeds from
"""
def get_liver_roi(ct_scan,aorta_seg,body_segmentation):

    #TODO - CHECK THIS!!!
    ct_scan_data = ct_scan.get_data()
    aorta_seg_data = aorta_seg.get_data()
    body_segmentation_data = body_segmentation.get_data()
    # first, let get all the new boundaries for the roi
    aorta_components_array = measure.label(aorta_seg_data)
    body_components_array = measure.label(body_segmentation_data)

    # axial box
    liver_roi_min_axial, liver_roi_max_axial,center_axial = get_axial_bounding_from_aorta_seg(aorta_components_array)

    # coronal box
    liver_roi_max_coronal = get_max_coronal_bounding_from_body_seg(body_components_array)
    liver_roi_min_coronal = get_min_coronal_bounding_from_aorta_seg(aorta_seg_data,center_axial)

    # sagital box
    liver_roi_min_sagital, liver_roi_max_sagital = get_sagital_bounderies_from_body_seg_slices(body_segmentation_data,
                                                                        liver_roi_min_axial,liver_roi_max_axial)

    # # lets get the new min and max values
    # new_min_sagital, new_min_corotal, new_min_axial, new_max_sagital, new_max_corotal, \
    # new_max_axial = get_new_min_and_max_for_liver_roi(props[0].bbox)

    # and now we can create the ROI of the liver
    mask = np.zeros(body_segmentation.shape, bool)
    mask[int(liver_roi_min_sagital):int(liver_roi_max_sagital), int(liver_roi_min_coronal):int(liver_roi_max_coronal),
    int(liver_roi_min_axial):int(liver_roi_max_axial)] = True

    liver_roi_img = np.where(mask == True, ct_scan_data, 0)
    liver_roi_segmentation = np.where(mask == True, 1, 0)
    # # first we will take
    # # the liver is at the lower coronal axis than the aorta. first we want to take only the roi of the body
    # body_roi = get_ROI_image_from_binary_ROI(ct_scan,body_segmentation)

    # now we want to get a more specific region of the liver - we want to take some box near the aorta, which will
    # contains the liver. we know that it's a little in front of the aorta, ao we will use this info
    # so lets get bbox limits of the liver, and then we will create the ROI

    # liver_roi = get_liver_roi_near_aorta(Aorta_seg, body_roi)

    return liver_roi_img,liver_roi_segmentation

"""
will take a file name and will create and return a nifty image object, and it's data array
"""
def load_nifty_file_and_get_data(file_name):
    img = nib.load(file_name)
    if nib.aff2axcodes(img.affine)!= ('R','A','S'):
        img = nib.as_closest_canonical(img)
    img_data = img.get_data()

    return img,img_data

"""
one of the main program functions. will get names of nifty images files, containing the original CT scan, and the 
aorta segmentation, and will return the segmentation of the liver - by using the other methods we wrote, implementing 
the RG technique
"""
def segmentLiver(ctFileName, AortaFileName, outputFileName):
    # first lets load all our inputs in order to be able to use them
    ct_scan_for_body_seg = load_nifty_file_and_get_data(ctFileName)[0]
    ct_scan,ct_scan_data = load_nifty_file_and_get_data(ctFileName)
    aorta_seg = load_nifty_file_and_get_data(AortaFileName)[0]

    # next, we want to get the body segmentation
    IsolateBody(ct_scan_for_body_seg)

    # now lets get the liver ROI
    liver_roi_seg = get_liver_roi(ct_scan,aorta_seg,ct_scan_for_body_seg)[1]

    # now we will call the region growing method, which will make the ct scan object to the corresponding segmentation
    seg_array = multipleSeedsRegionGrowing(ct_scan,liver_roi_seg)

    # seg_array = alternative_region_growing(ct_scan,liver_roi_seg)

    # # now lets save this as a segmentation
    # indexes = tuple(np.array(list(zip(*region_pixels))))
    # mask = np.zeros(ct_scan.shape)
    # mask[indexes] = 1
    # ct_scan_data[True] = mask
    ct_scan_data[True] = seg_array
    nib.save(ct_scan, outputFileName)
    return

#########################comparing the truth and the estimation######################################################

"""
 calculates the dice coefficient of an estimated image and a ground truth one
"""
def calc_dice_coefficient(estimated_img_data, num_of_intersect_voxels, truth_img_data):
    num_of_voxels_in_the_truth = truth_img_data.sum()
    num_of_voxels_int_the_estimate = estimated_img_data.sum()
    dice_coefficient = (2 * num_of_intersect_voxels) / (num_of_voxels_in_the_truth + num_of_voxels_int_the_estimate)
    return dice_coefficient

"""
 this function will get estimated and truth seg images, and will calculate which how much they have in common
"""
def get_intersection_and_union_data(estimated_img_data, truth_img_data):
    intersection_img = truth_img_data.astype(np.uint64) & estimated_img_data.astype(np.uint64)
    num_of_intersect_voxels = intersection_img.sum()
    union_img = truth_img_data.astype(np.uint64) | estimated_img_data.astype(np.uint64)
    num_of_union_voxels = union_img.sum()
    return num_of_intersect_voxels, num_of_union_voxels

def get_sum_of_truth_pixels_from_estimation(surface_estimation_pixels,surface_truth_img_pixels):
    sum_of_truth_pixels_from_estimation = 0
    for truth_pixel in surface_truth_img_pixels:
        sum_of_truth_pixels_from_estimation += min([math.sqrt((truth_pixel[0] - estimation_pixel[0]) ** 2 +
                                                              (truth_pixel[1] - estimation_pixel[1]) ** 2 +
                                                              (truth_pixel[2] - estimation_pixel[2]) ** 2) for
                                                    estimation_pixel in
                                                    surface_estimation_pixels])
    return sum_of_truth_pixels_from_estimation

def get_sum_of_estimation_pixels_from_truth(surface_estimation_pixels,surface_truth_img_pixels):
    sum_of_estimation_pixels_from_truth = 0
    for estimation_pixel in surface_estimation_pixels:
        sum_of_estimation_pixels_from_truth += min([math.sqrt((estimation_pixel[0] - truth_pixel[0]) ** 2 +
                                                              (estimation_pixel[1] - truth_pixel[1]) ** 2 +
                                                              (estimation_pixel[2] - truth_pixel[2]) ** 2) for
                                                    truth_pixel in
                                                    surface_truth_img_pixels])
    return sum_of_estimation_pixels_from_truth

"""
 the ASSD is a surface distance calculation (as oppose to the other 2 who are volume based)
 the ASSD computes the sum of distance between each point in one surface to all the points in the other
"""
def calc_assd_distance(truth_img_data, estimated_img_data):
    # first - we will get both surfaces points, get their size and calculate all possible min distances
    estimated_img_seg_pixels = measure.regionprops(measure.label(estimated_img_data))[0].coords
    truth_img_seg_pixels = measure.regionprops(measure.label(truth_img_data))[0].coords
    surface_estimation_pixels = [pixel for pixel in estimated_img_seg_pixels if not np.all(estimated_img_data[pixel[0]-1:pixel[0]+2,
                                                                                pixel[1]-1:pixel[1]+2,pixel[2]-1:pixel[2]+2])]
    surface_truth_img_pixels = [pixel for pixel in truth_img_seg_pixels if not np.all(truth_img_data[pixel[0]-1:pixel[0]+2,
                                                                                pixel[1]-1:pixel[1]+2,pixel[2]-1:pixel[2]+2])]

    # then, we will get 2 sums - sum of min distances between every point in one surface the other surface

    # 1. for every pixel in the estimation surface, we will get the min distance from the truth surface by measuring is
    #    from all the truth surface pixels. then - we will sum up the results of all estimation pixels. do it both directions
    sum_of_estimation_pixels_from_truth = get_sum_of_estimation_pixels_from_truth(surface_estimation_pixels,surface_truth_img_pixels)
    sum_of_truth_pixels_from_estimation = get_sum_of_truth_pixels_from_estimation(surface_estimation_pixels,surface_truth_img_pixels)

    # 2. now lets get the final value
    final_assd = 0.5*((sum_of_estimation_pixels_from_truth/len(surface_estimation_pixels))+
                      (sum_of_truth_pixels_from_estimation/len(surface_truth_img_pixels)))

    return final_assd

"""
  this function will get the ground truth segmentation and the estimated one, and will calculate the volume difference
  the dice coefficient, and the ASSD.
  I assume the input for this function is file names - a name of 2 nifty files representing the segmentation and the 
  ground truth.
  the function will return only the VOD
"""
def evaluateSegmentation(ground_truth_segmentation, estimated_segmentation):
    # first we will get arrays from the data, which we can work on
    truth_img,truth_img_data = load_nifty_file_and_get_data(ground_truth_segmentation)
    truth_img_data = truth_img_data.astype(bool)

    # truth_img = nib.load(ground_truth_segmentation)
    # truth_img_data =truth_img.get_data().astype(bool)
    estimated_img,estimated_img_data = load_nifty_file_and_get_data(estimated_segmentation)
    # estimated_img = nib.load(estimated_segmentation)
    estimated_img_data = estimated_img_data.astype(bool)

    # now we want to get the images intersection and union
    num_of_intersect_voxels, num_of_union_voxels = get_intersection_and_union_data(estimated_img_data, truth_img_data)

    # lets get the volume overlap difference
    volume_overlap_difference = 1-(num_of_intersect_voxels/num_of_union_voxels)
    # lets get the dice coefficient
    dice_coefficient = calc_dice_coefficient(estimated_img_data, num_of_intersect_voxels, truth_img_data)
    # lets get the ASSD
    ASSD = calc_assd_distance(truth_img_data,estimated_img_data)

    return volume_overlap_difference



def test_func(name_of_scan,seg_name):

    nifty_image,nifty_image_data = load_nifty_file_and_get_data(name_of_scan)
    # nifty_image = nib.load(name_of_scan)
    # nifty_image = nib.as_closest_canonical(nifty_image)
    # nib.save(nifty_image,"case1_flipped.nii.gz")
    # nifty_image_data = nifty_image.get_data()
    seg_image,seg_image_data = load_nifty_file_and_get_data(seg_name)
    # seg_image = nib.load(seg_name)
    # seg_image = nib.as_closest_canonical(seg_image)
    # seg_image_data = seg_image.get_data()
    body_seg,body_seg_data = load_nifty_file_and_get_data(name_of_scan)
    # body_seg = nib.load(name_of_scan)
    # body_seg = nib.as_closest_canonical(body_seg)

    # isolate body test
    IsolateBody(body_seg)
    # nib.save(nifty_image,get_name_to_save_nifty_file(name_of_scan,"_body_scan"))
    # img = nib.load("Case2_CT_body_scan.nii.gz")
    # print(measure.label(img.get_data(), return_num=True)[1])

    # # find seeds test
    # arr = np.zeros(nifty_image.shape)
    # arr[:,:,6:9] = 1
    # print(find_seeds(nifty_image,arr))

    # get liver roi
    liver_roi_img,liver_roi_seg = get_liver_roi(nifty_image,seg_image,body_seg)
    to_save = nib.Nifti1Image(liver_roi_img,nifty_image.affine,nifty_image.header)
    nib.save(to_save,get_name_to_save_nifty_file(name_of_scan,"_liver_roi"))

    liver_roi_seed = get_liver_roi_for_seeds_search(liver_roi_seg)
    to_save2 = nib.Nifti1Image(get_ROI_image_from_binary_ROI(nifty_image,liver_roi_seed),nifty_image.affine,nifty_image.header)
    nib.save(to_save2,get_name_to_save_nifty_file(name_of_scan,"_liver_seed_roi"))

    # get liver segmentation

    return
#
# evaluateSegmentation("Case1_liver_segmentation.nii.gz","Case1_CT_MyLiverSeg.nii.gz")
# segmentLiver("HardCase1_CT.nii.gz", "HardCase1_Aorta.nii.gz", "HardCase1_CT_MyLiverSeg.nii.gz")

# test_func("HardCase1_CT.nii.gz","HardCase1_Aorta.nii.gz")
# img = nib.load("HardCase1_CT.nii.gz")
# if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
#     img = nib.as_closest_canonical(img)
# nib.save(img,"HardCase1_good_direction.nii.gz")

# img = nib.load("HardCase1_liver_segmentation.nii.gz")
# if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
#     img = nib.as_closest_canonical(img)
# nib.save(img,"liver_seg_HardCase1_good_direction.nii.gz")

# x,y,z = 2,3,4
# all_neighbors = [Pixel((x + i, y + j, z + k)) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
# img = nib.load("Case1_CT.nii.gz")
# img_data = img.get_data()
# out_img = np.zeros(img_data.shape)
# x,y,z = 7,7,7
# arr = np.zeros(img_data.shape)
# out_img[x-1:x+2,y-1:y+2,z-1:z+2] = np.where(img_data[x-1:x+2,y-1:y+2,z-1:z+2]<-100,1,0)
# print(out_img)
# print([i.location for i in all_neighbors])
# arr = [[[1],[2],[3]],[[4],[5],[99]]]
# indexes = [(0,0),(1,1),(0,2)]
# cond = arr[indexes]==1

# set = set(6)
# set.add(7)
# set.add(6)

# img_data = nib.load("Case1_CT_MyLiverSeg.nii.gz").get_data()
# print(77)
# for (x,y,z),val in np.ndenumerate(img_data):
#     img_data[int(x),int(y),int(z)] = 77


