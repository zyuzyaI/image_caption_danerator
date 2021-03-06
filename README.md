Image caption generator is a project to recognize the context of an image and describe them in a natural language like English.

The objective of the project is build a working model of Image caption generator by implementing CNN with LSTM.

For the image caption generator, we will be using the Flickr_8K dataset. 

[Flicker8k_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
[Flickr_8k_text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

Files in the project.

* Models – contain our trained models.
* Descriptions.txt – text file contains all image names and their captions after preprocessing.
* Features.p – Pickle object that contains an image and their feature vector extracted from the Xception pre-trained CNN model.
* Tokenizer.p – Contains tokens mapped with an index value.
* Model.png – Visual representation of dimensions of our project.
* Testing_caption_generator.py – Python file for generating a caption of any image.
* Training_caption_generator.ipynb – Jupyter notebook in which we train and build our image caption generator.