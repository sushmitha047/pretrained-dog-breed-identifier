#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py
#
# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: H S Sushmitha
# DATE CREATED: 11/11/2023                               
# REVISED DATE: 11/11/2023
# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task. 
#          Note that the true identity of the pet (or object) in the image is 
#          indicated by the filename of the image. Therefore, your program must
#          first extract the pet image label from the filename before
#          classifying the images using the pretrained CNN model. With this 
#          program we will be comparing the performance of 3 different CNN model
#          architectures to determine which provides the 'best' classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from time import time, sleep

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

def save_final_results(results_stats_resnet, results_stats_alexnet, results_stats_vgg):
    with open('final_results.txt', 'w') as f:
        f.write("Final Comparison of Model Architectures\n")
        f.write("=====================================\n\n")
        
        models = ['ResNet', 'AlexNet', 'VGG']
        stats = [results_stats_resnet, results_stats_alexnet, results_stats_vgg]
        
        for model, stat in zip(models, stats):
            f.write(f"{model} Results:\n")
            f.write(f"N Images: {stat['n_images']}\n")
            f.write(f"N Dog Images: {stat['n_dogs_img']}\n")
            f.write(f"N Not-Dog Images: {stat['n_notdogs_img']}\n")
            f.write(f"pct_match: {stat['pct_match']:.1f}%\n")
            f.write(f"pct_correct_dogs: {stat['pct_correct_dogs']:.1f}%\n")
            f.write(f"pct_correct_breed: {stat['pct_correct_breed']:.1f}%\n")
            f.write(f"pct_correct_notdogs: {stat['pct_correct_notdogs']:.1f}%\n\n")


# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    
    # TODO 1: Define get_input_args function within the file get_input_args.py
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg  
    check_command_line_arguments(in_arg)

    
    # TODO 2: Define get_pet_labels function within the file get_pet_labels.py
    # Once the get_pet_labels function has been defined replace 'None' 
    # in the function call with in_arg.dir  Once you have done the replacements
    # your function call should look like this: 
    #             get_pet_labels(in_arg.dir)
    # This function creates the results dictionary that contains the results, 
    # this dictionary is returned from the function call as the variable results
    results = get_pet_labels(in_arg.dir)

    # Function that checks Pet Images in the results Dictionary using results    
    check_creating_pet_image_labels(results)


    # TODO 3: Define classify_images function within the file classiy_images.py
    # Once the classify_images function has been defined replace first 'None' 
    # in the function call with in_arg.dir and replace the last 'None' in the
    # function call with in_arg.arch  Once you have done the replacements your
    # function call should look like this: 
    #             classify_images(in_arg.dir, results, in_arg.arch)
    # Creates Classifier Labels with classifier function, Compares Labels, 
    # and adds these results to the results dictionary - results
    classify_images(in_arg.dir, results, in_arg.arch)

    # Function that checks Results Dictionary using results    
    check_classifying_images(results)    

    
    # TODO 4: Define adjust_results4_isadog function within the file adjust_results4_isadog.py
    # Once the adjust_results4_isadog function has been defined replace 'None' 
    # in the function call with in_arg.dogfile  Once you have done the 
    # replacements your function call should look like this: 
    #          adjust_results4_isadog(results, in_arg.dogfile)
    # Adjusts the results dictionary to determine if classifier correctly 
    # classified images as 'a dog' or 'not a dog'. This demonstrates if 
    # model can correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(results, in_arg.dogfile)

    # Function that checks Results Dictionary for is-a-dog adjustment using results
    check_classifying_labels_as_dogs(results)


    # TODO 5: Define calculates_results_stats function within the file calculates_results_stats.py
    # This function creates the results statistics dictionary that contains a
    # summary of the results statistics (this includes counts & percentages). This
    # dictionary is returned from the function call as the variable results_stats    
    # Calculates results of run and puts statistics in the Results Statistics
    # Dictionary - called results_stats
    results_stats = calculates_results_stats(results)

    # Function that checks Results Statistics Dictionary using results_stats
    check_calculating_results(results, results_stats)


    # TODO 6: Define print_results function within the file print_results.py
    # Once the print_results function has been defined replace 'None' 
    # in the function call with in_arg.arch  Once you have done the 
    # replacements your function call should look like this: 
    #      print_results(results, results_stats, in_arg.arch, True, True)
    # Prints summary results, incorrect classifications of dogs (if requested)
    # and incorrectly classified breeds (if requested)
    print_results(results, results_stats, in_arg.arch, True, True)
    
#     #checking TODO 0:
#     sleep(10)
    
    # TODO 0: Measure total program runtime by collecting end time
    end_time = time()

    # Run for ResNet
    results_resnet = get_pet_labels(in_arg.dir)
    classify_images(in_arg.dir, results_resnet, 'resnet')
    adjust_results4_isadog(results_resnet, in_arg.dogfile)
    results_stats_resnet = calculates_results_stats(results_resnet)
    print_results(results_resnet, results_stats_resnet, 'resnet', True, True)
    
    # Run for AlexNet
    results_alexnet = get_pet_labels(in_arg.dir)
    classify_images(in_arg.dir, results_alexnet, 'alexnet')
    adjust_results4_isadog(results_alexnet, in_arg.dogfile)
    results_stats_alexnet = calculates_results_stats(results_alexnet)
    print_results(results_alexnet, results_stats_alexnet, 'alexnet', True, True)
    
    # Run for VGG
    results_vgg = get_pet_labels(in_arg.dir)
    classify_images(in_arg.dir, results_vgg, 'vgg')
    adjust_results4_isadog(results_vgg, in_arg.dogfile)
    results_stats_vgg = calculates_results_stats(results_vgg)
    print_results(results_vgg, results_stats_vgg, 'vgg', True, True)
    
    # Save final comparison
    save_final_results(results_stats_resnet, results_stats_alexnet, results_stats_vgg)
    
    # TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"+str(int((tot_time%3600)%60)) )



        
# Call to main function to run the program
if __name__ == "__main__":
    main()
