###################################################################
# FILE : ex6.py
# WRITER : HANIT_HAKIM , HANIT1992 , 308396480
# EXERCISE : intro2cs ex6 2015-2016
# DESCRIPTION : A program that makes a new image from a given one,
#constructed from a lot of small pictures.
###################################################################

import mosaic
import sys
NUMBER_OF_ARGUMENTS=5
import copy

def compare_pixel(pixel1, pixel2):
    """this function gets a couple of two different pixels, and will return
    the sum of the differnce between each of the color values for the two pixels"""
    (r1,g1,b1)=pixel1
    (r2,g2,b2)=pixel2
    r_distance=r1-r2
    if r_distance<0:
        r_distance=-1*r_distance
    g_distance=g1-g2
    if g_distance<0:
        g_distance=-1*g_distance
    b_distance=b1-b2
    if b_distance<0:
        b_distance=-1*b_distance
    the_pixels_distance=r_distance + g_distance + b_distance

    return the_pixels_distance



def compare(image1, image2):
    """this function will compare all the pixel's distances of the two picturs,
     according to the common pixels of them"""
    pictures_distance_list=[]
    for line_of_pixels_spot in range(len(image1)):
        for row_of_pixels_spot in range(len(image1[line_of_pixels_spot])):
            try:
                image2[line_of_pixels_spot][row_of_pixels_spot]
            except IndexError:
                break
            else:
                image1_pixel=image1[line_of_pixels_spot][row_of_pixels_spot]
                image2_pixel=image2[line_of_pixels_spot][row_of_pixels_spot]
                pictures_distance_list.append(compare_pixel(image1_pixel,image2_pixel))
    distance_between_pictures=sum(pictures_distance_list)
    return distance_between_pictures

def get_piece(image, upper_left, size):
    """this function will get a specific piece from the picture and will create
    a new minimize one, represented by a list of lists"""
    pixel_list_for_new_picture=[]
    row=upper_left[0]
    column=upper_left[1]
    hight=size[0]
    weight=size[1]
    try:
        image[row:row+hight]
    except IndexError:
        pixels_list_for_row=image[row: ]
    else:
        pixels_list_for_row=image[row:row+hight]
    for a_row in pixels_list_for_row:
        try:
            a_row[column:column+weight]
        except IndexError:
            a_row=a_row[column: ]
        else:
            a_row=a_row[column:column+weight]
        pixel_list_for_new_picture.append(a_row)
    return pixel_list_for_new_picture

def set_piece(image, upper_left, piece):
    """this function will set a new image in a chosen spot of the
    original image"""
    size_of_new_img=(len(piece),len(piece[0]))
    list_for_new_img_place=get_piece(image,upper_left,size_of_new_img)
    top_row=upper_left[0]
    top_column=upper_left[1]
    for row in range(len(list_for_new_img_place)):
        for column in range(len(list_for_new_img_place[0])):
            image[top_row+row][top_column+column]=piece[row][column]

def average(image):
    """this function will calculate the average value of all the pixels of an
    image, defined by the colors that builds each pixel"""
    red_list=[]
    green_list=[]
    blue_list=[]
    for line in image:
        for row in line:
            
                red_list.append(row[0])
                green_list.append(row[1])
                blue_list.append(row[2])

    avarage_red=sum(red_list)/len(red_list)
    avarage_green=sum(green_list)/len(green_list)
    avarage_blue=sum(blue_list)/len(blue_list)
    return (avarage_red,avarage_green,avarage_blue)

def preprocess_tiles(tiles):
    """this function will take a list of images and return a list of the average
    pixels value for each of them"""
    avarage_images_list=[]
    for an_image in tiles:
        an_image=average(an_image)
        avarage_images_list.append(an_image)
    return avarage_images_list

def get_best_tiles(objective, tiles, averages , num_candidates):
    """this function will return a list of the best matching tiles
    for a certain part of the original picture"""
    compared_tiles_values=[]
    avarage_pixel_objective=average(objective)
    for tile_avarage_pixel_list in averages:
        compared_tiles_values.append(compare_pixel(tile_avarage_pixel_list,avarage_pixel_objective))
        list_of_chosen_candidates=[]
    for num in range(num_candidates):
        minimum=min(compared_tiles_values)
        list_of_chosen_candidates.append(tiles[compared_tiles_values.index(minimum)])
        compared_tiles_values[compared_tiles_values.index(minimum)]=766
        if compared_tiles_values==[]:
            break
    return list_of_chosen_candidates


def choose_tile(piece, tiles):
    """this function will choose the best tile match for a certain
    part of the original picture"""
    list_of_distances_for_possible_pictures=[]
    for picture in tiles:
        list_of_distances_for_possible_pictures.append(compare(picture,piece))
    the_picture_distance=min(list_of_distances_for_possible_pictures)
    the_chosen_picture=tiles[list_of_distances_for_possible_pictures.index(the_picture_distance)]

    return the_chosen_picture

def make_mosaic(image, tiles, num_candidates):
    """this function will generate a new photo,
    constructed by small photos that will create the closest outcome to
    the original picture, according to the pixels"""

    average_tiles_list=preprocess_tiles(tiles)
    copy_img = copy.deepcopy(image)
    for piece_in_row in range(0,len(image), len(tiles[0])):
        for piece_in_line in range(0, len(image[0]), len(tiles[0][0])):
            upper_left=(piece_in_row,piece_in_line)
            picture_by_piece=get_piece(image,upper_left,(len(tiles[0]),len(tiles[0][0])))
            list_of_best_tiles=get_best_tiles(picture_by_piece,tiles,average_tiles_list,num_candidates)
            chosen_tile=choose_tile(picture_by_piece,list_of_best_tiles)
            set_piece(copy_img,upper_left,chosen_tile)

    return copy_img


if __name__=="__main__":
    if len(sys.argv)==NUMBER_OF_ARGUMENTS+1:
        script_name= sys.argv[0]
        image_filename= sys.argv[1]
        images_dir= sys.argv[2]
        output_name= sys.argv[3]
        tile_height= int(sys.argv[4])
        num_candidates= int(sys.argv[5])
        image=mosaic.load_image(image_filename)
        tiles=mosaic.build_tile_base(images_dir,tile_height)
        final_picture=make_mosaic(image,tiles,num_candidates)
        mosaic.save(final_picture,output_name)
    else:
        print('hi there! this is not the correct number of paramenters. in order to '
              'run the program, the structure of your command should be:'
              'ex6.py <image_source> <images_dir> <output_name> <tile_height> <num_candidates>')





