import os, os.path
import shutil
import cv2

import constants
import kmeans
import drDill
import imgdirreader
import classifiers.bruteforce as bruteforce
import classifiers.histdiff as histdiff

CLASSIFIER_ALG_BF = 1
CLASSIFIER_ALG_HIST = 2

def classify(query_path, training_db, output_dir, results_prefix, classify_mode_alg=CLASSIFIER_ALG_BF):
    if training_db == "":
        print "A training data file should be specified (specify using -d or --data)"
        return

    kps, descs, threshold, centroids, hist = drDill.load_training_data(training_db)
    if len(descs) == 0:
        print "No training data loaded"
        return
        
    if classify_mode_alg == CLASSIFIER_ALG_BF:
        min_num_hits = len(descs)/3
        if min_num_hits == 0:
            min_num_hits = 1
        print "Images must have at least " + str(min_num_hits) + " hits from training set for a positive classification"
        
    if os.path.isdir(query_path):
        if output_dir == "":
            output_dir = os.path.normpath(query_path) + "_out"
            
        if os.path.exists(output_dir):
            ## Delete its contents!
            print "Removing " + output_dir + " prior to classification..."
            shutil.rmtree(output_dir)
        
        output_dir_yes = os.path.join(output_dir, "yes")
        output_dir_no = os.path.join(output_dir, "no")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_dir_yes):    
            os.makedirs(output_dir_yes)
        if not os.path.exists(output_dir_no):    
            os.makedirs(output_dir_no)

        files_to_proc = imgdirreader.get_image_files(query_path)
        if len(files_to_proc) == 0:
            print "Found no image files in " + query_path
            return
    else:
        if not os.path.exists(query_path):
            print "File not found: " + query_path
            return
        elif not imgdirreader.is_image_file(query_path):
            print query_path + " is not a recognized image file."
            return
        else:
            files_to_proc = [query_path]

    #surf = cv2.SURF(hessian, upright=0)            
    sift = cv2.SIFT()            
    bf = cv2.BFMatcher(cv2.NORM_L2)
            
    for qimg_pathname in files_to_proc:
        print "\n-------------------------------------\nClassifying " + qimg_pathname
        _, qimg_filename = os.path.split(qimg_pathname)    
        
        qimg = cv2.imread(qimg_pathname, 0)
        qimg = cv2.resize(qimg, (constants.RESCALE_WIDTH, constants.RESCALE_HEIGHT))
                                
        ## Get the descriptors of the query image
        _, qdescs = sift.detectAndCompute(qimg, None)
        print "Found " + str(len(qdescs)) + " descriptors in query image"
    
        yes_classify = False
        if classify_mode_alg == CLASSIFIER_ALG_BF:
            yes_classify = bruteforce.classify(qimg_pathname, qdescs, descs, bf, min_num_hits)
        elif classify_mode_alg == CLASSIFIER_ALG_HIST:
            yes_classify = histdiff.classify(qimg_pathname, qdescs, threshold, centroids, hist)
        
        if os.path.isdir(query_path):
            dest_path = os.path.join(output_dir_no, qimg_filename)
            if yes_classify:
                dest_path = os.path.join(output_dir_yes, qimg_filename)
            
            shutil.copyfile(qimg_pathname, dest_path)
        
    if results_prefix != "" and os.path.isdir(query_path):
        generate_confusion_matrix(output_dir_yes, output_dir_no, results_prefix)
    
def generate_confusion_matrix(output_dir_yes, output_dir_no, prefix):
    yes_yes, yes_no = count_filename_matches(output_dir_yes, prefix)
    no_yes, no_no = count_filename_matches(output_dir_no, prefix)
    
    print "\n-=-=-=-=-=- CLASSIFICATION RESULTS -=-=-=-=-=-"
    print "Results were computed by checking if the filename starts with \"" + prefix + "\""
    print "Column headers represent the truth, row headers represent the predictions done by the classifier"
    print "%8s" % "YES" + "%5s" % "NO"
    print "YES %4s" % str(yes_yes) + "%5s" % str(yes_no)
    print "NO %5s" % str(no_yes) + "%5s" % str(no_no)
    
    ## Compute accuracy
    total = yes_yes + yes_no + no_yes + no_no
    correct = yes_yes + no_no
    correct_rate = (correct/float(total))
    error_rate = 1 - correct_rate
    print "\nCorrectly classified: " + str(correct) + "/" + str(total) + " (%.2f" % (correct_rate * 100.0) + "%% accuracy, %.2f" % (error_rate * 100.0) + "% error rate)"
    return error_rate
            
def count_filename_matches(dir, prefix):
    yes_match = 0
    no_match = 0
    ## Get list of filenames in the given directory
    pathnames = imgdirreader.get_image_files(dir)
    ## For each filename, check if it starts with the given prefix,
    ## and increment counters based on the result
    for pname in pathnames:
        _, fname = os.path.split(pname)
        if fname.lower().startswith(prefix):
            yes_match += 1
        else:
            no_match += 1
    
    return yes_match, no_match