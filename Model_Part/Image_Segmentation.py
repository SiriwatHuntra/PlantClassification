from Util.BG_remove import Segment
# For normal model
Input = ["ClassificationModel/Image_2"]
Output = ["ClassificationModel/Image_Output_2"]

# For test noise tuffness of model
# Input = ["ClassificationModel/Noisy/blur", "ClassificationModel/Noisy/brightness", "ClassificationModel/Noisy/damage","ClassificationModel/Noisy/perspective","ClassificationModel/Noisy/speckle",]
# Output = ["ClassificationModel/Augmented_Out/blur", "ClassificationModel/Augmented_Out/brightness", "ClassificationModel/Augmented_Out/damage","ClassificationModel/Augmented_Out/perspective","ClassificationModel/Augmented_Out/speckle",]

for i, j in zip(Input, Output):
    Segment(i, j)
