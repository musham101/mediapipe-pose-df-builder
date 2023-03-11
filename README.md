# Pose Data Collection Script
This script allows you to collect pose data from images using the Mediapipe library and store them in a Pandas DataFrame. The generated dataset can be used for machine learning, computer vision, or any other application that requires pose analysis data.

# Requirements
- Python 3.x
- OpenCV 2 or higher
- Mediapipe library
# Installation
Clone or download this repository to your local machine.
Install the required packages by running the following command: `pip install -r requirements.txt`
# Usage
Place your images in a folder named `images` and change the folder_paths to `folder_paths = ['images']`.
Run the script by running the following command: `python pose_data_collection.py`
The script will generate a CSV file named `pose_data.csv` containing the pose data of all the images in the images folder.
# Contributing
Contributions are welcome! Please feel free to open a pull request or issue.

# License
This project is licensed under the MIT License.

# Acknowledgments
This project was inspired by the Mediapipe library and the need for a simple way to collect pose data from images.
