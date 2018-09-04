# CV-classify: image classification using OpenCV (SIFT module)

This is a fairly simple image classification Python script that I wrote as a project deliverable for my 2014 Machine Learning class at the Polytechnic University of Catalonia (UPC), back when I was doing the Master in Innovation and Research in Informatics program. It performs image classification using the SIFT (Scale-Invariant Feature Transform) module of the popular [OpenCV](https://opencv.org/) Python library. SIFT is a feature detection and description algorithm first published by David Lowe in 1999.

The classifier works by first reading a set of _training images_, from which features are detected and computed using SIFT. It then builds models and determines the parameters necessary for classification. Later the classifier can be executed to perform classification on a set of input images (also known as _query images_). Only images in JPG and PNG formats are accepted.

DISCLAIMER: It's been years since I wrote this code and my understanding of image classification is now fuzzier than it was even then, but I didn't want to lose it so I decided to put it up. If you are an OpenCV/machine learning expert and have some feedback/corrections, please let me know!

ADD'L CREDITS: The k-means clustering implementation (Lloyd's algorithm) was obtained from the wonderful [Data Science Lab](http://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/).

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

Let's say you want to train the classifier with a set of images of circles located in the directory `$HOME/cv-classify/training-circle`.

![Training images](/screenshots/training-images-circle-before.png)

You can do this by running

    $ python cv-classify -t $HOME/cv-classify/training-circle

If all goes well the training data file `cv-classify.train` should have been created in the same directory:

![Training images with training data](/screenshots/training-images-circle-after.png)

If you want to change the location of the training data file, use the `-o` or `--output` option to specify the target directory.

### Classifying images based on previous training

Now let's try to classify the images located in `$HOME/cv-classify/classify-circle`:

![Query images](/screenshots/query-images-circle.png)

To start classifying using the previously-generated training data:

    $ python cv-classify -c $HOME/cv-classify/classify-circle -d $HOME/cv-classify/training-circle/cv-classify.train
    
By default the classifier will use the directory `$HOME/cv-classify/classify-circle_out` (creating and/or deleting it if necessary) and then copy the images found in `$HOME/cv-classify/classify-circle` to either `$HOME/cv-classify/classify-circle_out/yes` (positive classification):

![Positive classification results](/screenshots/output-yes-circle.png)

or `$HOME/cv-classify/classify-circle_out/no` (negative classification):

![Negative classification results](/screenshots/output-no-circle.png)

depending on whether or not the classifier considered an input image as a "positive match" to its training data. 

### Verifying classification results

If the images to be classified have a prefix that identifies a successful match, the script can automatically verify the results of the classification and produce a confusion matrix for you. 

Going back to our example of circle images, if the filenames of the input images of circles are prefixed with "circle" (e.g. "circle1.jpg", "circle-blue.png"), we can verify the results of the classification using the `-r` or `--results` option:

    $ python cv-classify -c $HOME/cv-classify/classify-circle -d $HOME/cv-classify/training-circle/cv-classify.train -r circle

If all goes well you should output similar to the following:

    -=-=-=-=-=- CLASSIFICATION RESULTS -=-=-=-=-=-
    Results were computed by checking if the filename starts with "circle"
    Column headers represent the truth, row headers represent the predictions done by the classifier
         YES   NO
    YES    7    1
    NO     2    6
    
    Correctly classified: 13/16 (81.25% accuracy, 18.75% error rate)

According to this result, the classifier got it right 81.25% of the time. Not bad! 
 
## TODOs

Some ideas on possible improvements:

* Validate understanding of image classification to improve success rate
* Provide ability to tweak internal parameters (e.g. match ratio, k-means cluster sizes, etc) from command line
* More testing to improve handling of errors/edge cases 
* Make code more "pythonic" (I come from a Java background :))
* Implement other image classification algorithms (e.g. naive Bayes) and techniques (e.g. sliding window)