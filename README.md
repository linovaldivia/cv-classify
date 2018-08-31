# CV-classify: image classification using OpenCV (SIFT module)

This is a fairly simple image classification Python script that I wrote as a project deliverable for my Machine Learning class at the Polytechnic University of Catalonia (UPC), back when I was doing the Master in Innovation and Research in Informatics program in 2015. It performs image classification using the SIFT (Scale-Invariant Feature Transform) module of the popular [OpenCV](https://opencv.org/) Python library. SIFT is a feature detection and description algorithm first published by David Lowe in 1999.

The classifier works by first reading a set of _training images_, from which features are detected and computed using SIFT. It then builds models and determines the parameters necessary for classification. Later the classifier can be executed to perform classification on a set of input images (also known as _query images_). Only images in JPG and PNG formats are accepted.

Admittedly it's been years since I wrote this code and my understanding of image classification is now a bit fuzzy, but I didn't want to lose it so I decided to put it up. If you are an OpenCV/machine learning expert and have some feedback/corrections, please let me know!

## Requirements/Dependencies

This code was developed and tested on: 

* Python 2.7 with numpy 1.15.1
* OpenCV 2.4.13.6 (NOTE: OpenCV 3.x no longer has the SIFT/SURF modules included by default)

## Usage

Once all the dependencies have been installed, run:

    $ python cv-classify.py

To get a list of options:

    -t, --training <training-data-dir>       Enter training mode and use given directory for training data
    -c, --classify <img-dir>                 Enter classify mode, using the images in the given directory as input
    -v, --validate                           Perform validation (LOOCV) when clustering in training mode (default: false)
    -o, --output <output-dir>                Location of training data file (in training mode) or classified images (in classify mode) 
                                             (default: <training-data-dir>/cv-classify.train training mode, <img-dir>_out in classify mode)
    -a, --algorithm <classifier>             Use either "bf" (brute force) or "hist" (histogram) as the classifier algorithm (default: "bf")
    -d, --data <training-data-file>          Use given file as source of training data
    -r, --results <prefix-value>             [USE WITH -c] Check results after classification by inspecting filenames 
                                             (filename that starts with the given prefix means it should be classified as a positive)

## Examples

### Training the classifier

Let's say you want to train the classifier with a set of images located in the directory `$HOME/training_imgs`. You can do this by running

    $ python cv-classify -t $HOME/training_imgs

If all goes well the training data file `$HOME/training_imgs/cv-classify.train` should have been created. If you want to change the location of the training data file, use the `-o` or `--output` option to specify the target directory.

### Classifying images based on previous training

To start classifying images in `$HOME/imgs_to_classify` using the previously-generated training data:

    $ python cv-classify -c $HOME/imgs_to_classify -d $HOME/training_imgs/cv-classify.train
    
By default the classifier will use the directory `$HOME/imgs_to_classify_out` (creating it if necessary) and then copy the images found in `$HOME/imgs_to_classify` into either `$HOME/imgs_to_classify_out/yes` or `$HOME/imgs_to_classify_out/no`, depending on whether or not the classifier considered an input image as a "positive match", given its training data. 

### Verifying classification results

If the images to be classified have a prefix that identifies a successful match, the script can automatically verify the results of the classification and produce a confusion matrix for you. 

Let's say you want to check if the classifier can correctly identify images of circles, and the filenames of input images of circles are prefixed with "circle" (e.g. "circle1.jpg", "circle-blue.png"). To verify the results of the classification, use the `-r` or `--results` option:

    $ python cv-classify -c $HOME/imgs_to_classify -d $HOME/training_imgs/cv-classify.train -r circle

If all goes well you should output similar to the following:

    -=-=-=-=-=- CLASSIFICATION RESULTS -=-=-=-=-=-
    Results were computed by checking if the filename starts with "circle"
    Column headers represent the truth, row headers represent the predictions done by the classifier
         YES   NO
    YES    5    1
    NO     4    6
    
    Correctly classified: 11/16 (68.75% accuracy, 31.25% error rate)

According to this result, the classifier correctly identified an image as a circle 68.75% of the time. Not bad, but could use some improvement! 
 
## TODOs

Some ideas on possible improvements:

* Validate understanding of image classification to improve success rate
* Provide ability to tweak internal parameters (e.g. match ratio, k-means cluster sizes, etc) from command line
* More testing to improve handling of errors/edge cases 
* Make code more "pythonic" (I come from a Java background :))
* Implement other image classification algorithms (e.g. naive Bayes) and techniques (e.g. sliding window)