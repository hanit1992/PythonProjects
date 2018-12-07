# we will implement two main functions: a function who finds a liver segmentation, and a function who measures the
# success, comparing to some ground truth. but first, we will implement small functions which will help us with the
# implementation of those 2 main functions
import random

import nibabel as nib
import numpy as np
from skimage import measure,morphology
from Pixel import Pixel

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
ADDITION_TO_MIN_SAGITAL_SEED_ROI = 300
SUBSTRACTION_OF_MAX_CORONAL_SEED_ROI = 100
#TODO need to play with those 2 to get a full roi contains the liver in all situations, as a bbox to the region growing search
MAX_LIVER_CORONAL_SUBSTRACTION = 10
MIN_LIVER_CORONAL_ADDTION = -40
AXIAL_SLICE_ADDITION_TO_FIND_SEEDS = 2
AXIAL_SLICE_ADDITION = 30
MIN_LIVER_SAGITAL_ADDITION = 20
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

"""
this function will return the segmentation based on a randomly chose seeds from the given ROI, by using the region
growing segmentation method
"""
def multipleSeedsRegionGrowing(ct_scan, ROI):

    seeds_roi = get_liver_roi_for_seeds_search(ROI)
    full_roi_img = get_ROI_image_from_binary_ROI(ct_scan,ROI)
    # first, we want to get the seeds. we will get them randomly from the liver ROI we found
    seeds_of_region = find_seeds(ct_scan,seeds_roi)

    # now, when we have the seeds, we will start the algorithm.
    # we will have 3 lists, in all of them the initial elements will be the seeds - a list of all the pixels in the
    # region, a list of all the pixels to check their neighbors, and a list of all the pixels we already labeled

    pixels_in_region = seeds_of_region #setting the initial list with the seeds
    pixels_to_investigate = seeds_of_region # setting a list of all the pixels who are in the region,
                                                               # but we need to go through the neighbors
    labeled_pixels = seeds_of_region

    # lets get the ct image data
    ct_scan_data = ct_scan.get_data()
    components_array = measure.label(ROI)
    bbox = measure.regionprops(components_array)[0].bbox
    # entering a loop - as long as we steel have some pixels to check
    while len(pixels_to_investigate) > 0:
        curr_pixel = pixels_to_investigate.pop(0)
        curr_pixel_neighbors = get_26_neighbors(curr_pixel)
        indexes = tuple(np.array(list(zip(*pixels_in_region))))
        current_region_intensities = ct_scan_data[indexes]
        current_region_mean = np.mean(current_region_intensities)

        if set(curr_pixel_neighbors)<= set(labeled_pixels):
            continue

        indexes = tuple(np.array(list(zip(*curr_pixel_neighbors))))
        inside_range_neighbors = [pixel for pixel in curr_pixel_neighbors if (ct_scan_data[pixel]!=0 and ct_scan_data[pixel]
                                  <= current_region_mean+5 and ct_scan_data[pixel] >= current_region_mean-5 and pixel
                                  not in labeled_pixels and pixel not in pixels_to_investigate and ct_scan_data[pixel] <= 200 and ct_scan_data[pixel]>=-100
                                  and check_if_pixel_is_in_bbox_for_region_growing(bbox,pixel))]
        # curr_pixel_neighbors = np.array(curr_pixel_neighbors)
        # cond = (ct_scan_data[indexes]!=0) & (ct_scan_data[indexes]<=current_region_mean+3) &\
        #        (ct_scan_data[indexes]>=current_region_mean-3) & (curr_pixel_neighbors not in labeled_pixels) & \
        #        (ct_scan_data[indexes] <= 200) & (ct_scan_data[indexes] >= -100)\
        #                           & (check_if_pixel_is_in_bbox_for_region_growing(bbox, curr_pixel_neighbors))
        # inside_range_neighbors = curr_pixel_neighbors[cond]

        print(len(inside_range_neighbors))
        #
        # in_region_cond = np.logical_and(ct_scan_data[curr_pixel_neighbors] <= current_region_mean+20,
        #                                ct_scan_data[curr_pixel_neighbors] >= current_region_mean-20)
        # inside_range_neighbors = list(curr_pixel_neighbors[np.where(in_region_cond)])
        # #adding another condition
        # inside_range_neighbors = list(inside_range_neighbors[inside_range_neighbors not in labeled_pixels])
        # inside_range_neighbors = list(curr_pixel_neighbors[in_region_cond])
        pixels_in_region = pixels_in_region + inside_range_neighbors
        pixels_to_investigate = pixels_to_investigate + inside_range_neighbors
        labeled_pixels = labeled_pixels + curr_pixel_neighbors
        print(len(pixels_to_investigate))

    return pixels_in_region

# def get_new_min_and_max_for_liver_roi(aorta_bbox):
#     min_sagital, min_corotal, min_axial, max_sagital, max_corotal, max_axial = aorta_bbox
#     centeral_axial = (min_axial+max_axial)/2
#     new_min_axial = centeral_axial - ADDITION_TO_GET_LIVER_ROI
#     new_max_axial = centeral_axial + ADDITION_TO_GET_LIVER_ROI
#     new_min_sagital = min_sagital + ADDITION_TO_SAGITAL
#     new_max_sagital = max_sagital + ADDITION_TO_SAGITAL*2
#
#     return new_min_sagital,min_corotal,new_min_axial,new_max_sagital,max_corotal,new_max_axial
#
# def get_liver_roi_near_aorta(aorta_seg, body_roi):
#     conmponents_array = measure.label(aorta_seg)
#     props = measure.regionprops(conmponents_array)
#
#     # lets get the new min and max values
#     new_min_sagital, new_min_corotal, new_min_axial, new_max_sagital, new_max_corotal,\
#     new_max_axial = get_new_min_and_max_for_liver_roi(props[0].bbox)
#
#     # and now we can create the ROI of the liver
#     mask = np.zeros(body_roi.shape, bool)
#     mask[int(new_min_sagital):int(new_max_sagital), int(new_min_corotal):int(new_max_corotal),
#     int(new_min_axial):int(new_max_axial)] = True
#
#     liver_roi = np.where(mask == True, body_roi, 0)
#
#     return liver_roi

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
    region_pixels = multipleSeedsRegionGrowing(ct_scan,liver_roi_seg)

    # now lets save this as a segmentation
    indexes = tuple(np.array(list(zip(*region_pixels))))
    mask = np.zeros(ct_scan.shape)
    mask[indexes] = 1
    ct_scan_data[True] = mask
    nib.save(ct_scan, outputFileName)
    return

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
segmentLiver("Case1_CT.nii.gz", "Case1_Aorta.nii.gz", "Case1_CT_MyLiverSeg.nii.gz")

# test_func("HardCase1_CT.nii.gz","HardCase1_Aorta.nii.gz")
# img = nib.load("Case1_Aorta.nii.gz")
# if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
#     img = nib.as_closest_canonical(img)
# nib.save(img,"Aorta_Case1_good_direction.nii.gz")

# img = nib.load("HardCase1_liver_segmentation.nii.gz")
# if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
#     img = nib.as_closest_canonical(img)
# nib.save(img,"liver_seg_HardCase1_good_direction.nii.gz")

# x,y,z = 2,3,4
# all_neighbors = [Pixel((x + i, y + j, z + k)) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
#
# print([i.location for i in all_neighbors])
# arr = [[[1],[2],[3]],[[4],[5],[99]]]
# indexes = [(0,0),(1,1),(0,2)]
# cond = arr[indexes]==1
