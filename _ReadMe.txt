# Read-me for findwaldo #
Group: Simon Scheurer, Yves Jegge


The function findwaldo.py searches Waldo in a given image and gives x- and y-coordinate back of Waldo. Zero position is in the left bottom corner.
The function main.py provides a test for the function findwaldo.py. 


Needed Packages: numpy, matplotlib, cv2, skimage, scipy 

Python-Files: 	- findwaldo.py: 		=> requested function to find Waldo recieve RGB-Image and returns computed X,Y-Coordinate from Waldo
		- ColorMatching.py 		=> Implements function searching Waldo by Color
		- ShapeMatching.py 		=> Implements function searching Waldo by Shape
		- TemplateMatching.py 		=> Implements function searching Waldo by templates
		- FaceMatching.py		=> Implements function searching Waldo by face detection using haar cascades model
		- main.py			=> Implements our function for testing findwaldo.py
		- GenerateImages.py		=> Generate negative images for haar cascades model (only used for Training)
		- KeyPointMatching.py 		=> Testing a Key-Point matching algorithm to find waldo (not used for find Waldo)


File Structure:	- findwaldo.py
				- ColorMatching.py
				- ShapeMatching.py 
				- TemplateMatching.py
				- FaceMatching.py
				- main.py
				- GenerateImages.py
				- KeyPointMatching.py 
				- data
					- ground_truths: provided Dataset
					- images_1: provided Dataset
					- images_2: found Dataset
					- haarcascades: generated haarcascades models
					- templates: generated templates
					- waldo: waldo cut outs
					