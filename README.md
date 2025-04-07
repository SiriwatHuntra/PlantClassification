# PlantClassification
This is my project: leaf image base plant classificatin.
Pubic version, no webapp part included

# Flow 
  - Image Read
  - Image Processing: feature vector extraction
  - Port data into CSV file
  - Data Visualization
  - Model Making with Model selection/Comparation
  - Model evaluation

# File structure
In input folder path should contain with sub folder inside for sameple: 
  -Plant_Image
  -- →Class_A
  -- →Class_B
Output of feature extraction will automatic labeling by sub folder names
  
# Python File
  Extract
    - _ExtractionToCSV.py: Model Feature Extraction Pipeline, Input should be folder and it will return with CSV file
    - Feature.py: Feature vector extractor
  Util
    - BG_remove.py: Call Segment function to remove bg, input and out put is folder path
                  Usage sample is in "Image_segmentation.py"
    - Image_IO.py: incude with file IO and other utility function such as file format converter
  Image_Segmentation.py: Main file to call functuion from Util folders
  MakeModel.py: Make, Compare and select model for usage, include with model auto fine tune and evaluation
  NoiseAdding.py: Use for add noise into photo dataset
  RobustTest.py: Compare how efficient and accuracy of model on noisy dataset that generate from NoiseAdding.py
  Visual.py: use for visualize CSV data with Box plot and data overview
