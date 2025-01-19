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

3. Run individual models:
```
   python check_images.py --dir pet_images/ --arch [architecture] --dogfile dognames.txt
```
   Replace `[architecture]` with `resnet`, `alexnet`, or `vgg`.

4. Or run all models at once:
```
   sh run_models_batch.sh
```

## Project Results
After evaluating three CNN architectures on 40 images (30 dog images and 10 non-dog images), here are the key findings:

| Model   | Overall Match | Dog Detection | Breed Accuracy | Non-Dog Detection |
|---------|---------------|---------------|----------------|-------------------|
| VGG     | 87.5%        | 100%          | 93.3%         | 100%             |
| ResNet  | 82.5%        | 100%          | 90.0%         | 90.0%            |
| AlexNet | 75.0%        | 100%          | 80.0%         | 100%             |

VGG architecture demonstrated the best performance with:
- Highest overall match rate (87.5%)
- Perfect dog detection accuracy (100%)
- Best breed classification accuracy (93.3%)
- Perfect non-dog detection (100%)

## Acknowledgments
- AWS AI/ML Scholarship Program
- Udacity
