# Pre-trained Image Classifier to Identify Dog Breeds

## Project Overview
This project is part of the AWS AI/ML Scholarship Program Nanodegree - AI Programming with Python. The goal is to use a pre-trained image classifier to identify dog breeds and evaluate the performance of different CNN architectures.

## Project Objectives
- Accurately identify dog images vs. non-dog images.
- Correctly classify dog breeds in dog images.
- Determine the best CNN model architecture (ResNet, AlexNet, or VGG) for these tasks.
- Analyze the time-accuracy trade-off for each model.

## Project Structure
- `check_images.py`: Main script for the image classification process.
- `get_input_args.py`: Handles command line arguments.
- `get_pet_labels.py`: Extracts pet image labels from filenames.
- `classify_images.py`: Classifies pet images using the specified CNN model.
- `adjust_results4_isadog.py`: Adjusts the results dictionary to indicate whether labels are of dogs.
- `calculates_results_stats.py`: Calculates results statistics.
- `print_results.py`: Prints a summary of the results, including misclassifications.
- `print_functions_for_lab_checks.py`: Contains print functions for lab checks.
- `dognames.txt`: Text file containing dog names used by the classifier and found in pet image files.

## How to Run
1. Clone the repository:
```
   git clone https://github.com/sushmitha047/pretrained-dog-breed-identifier.git
   cd pretrained-dog-breed-identifier
```

2. Install the required dependencies:
```
   pip install -r requirements.txt
```

3. Ensure all required files are in the project directory.

4. Run the program using the command:
```
   python check_images.py --dir pet_images/ --arch [architecture] --dogfile dognames.txt
```
   Replace `[architecture]` with `resnet`, `alexnet`, or `vgg`.

5. Repeat for each architecture and compare results.

6. Or run the program all at once using the below command:
```
   sh run_models_batch.sh
```

## Acknowledgments
- AWS AI/ML Scholarship Program
- Udacity
