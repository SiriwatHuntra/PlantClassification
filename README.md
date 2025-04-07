# PlantClassification
This is my project: leaf image base plant classificatin.<br/>
Pubic version, no webapp part included<br/>

# Flow 
  - Image Read
  - Image Processing: feature vector extraction
  - Port data into CSV file
  - Data Visualization
  - Model Making with Model selection/Comparation
  - Model evaluation

# File structure
In input folder path should contain with sub folder inside for sameple:<br/>
  -Plant_Image<br/>
  -→Class_A<br/>
  -→Class_B<br/>
Output of feature extraction will automatic labeling by sub folder names
  
# Python File
  -Extract<br/>
    -_ExtractionToCSV.py: Model Feature Extraction Pipeline, Input should be folder and it will return with CSV file<br/>
    -Feature.py: Feature vector extractor<br/>
  -Util<br/>
    - BG_remove.py: Call Segment function to remove bg, input and out put is folder path<br/>
                  Usage sample is in "Image_segmentation.py"<br/>
    - Image_IO.py: incude with file IO and other utility function such as file format converter<br/>
  -Image_Segmentation.py: Main file to call functuion from Util folders<br/>
  -MakeModel.py: Make, Compare and select model for usage, include with model auto fine tune and evaluation<br/>
  -NoiseAdding.py: Use for add noise into photo dataset<br/>
  -RobustTest.py: Compare how efficient and accuracy of model on noisy dataset that generate from NoiseAdding.py<br/>
  -Visual.py: use for visualize CSV data with Box plot and data overview<br/>
