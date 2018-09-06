import sys
import getopt

import trainer
import classifier
            
def show_help():
    print "options: "
    print "-t, --training <training-data-dir>       Enter training mode and use given directory for training data"
    print "-c, --classify <img-dir>                 Enter classify mode, using the images in the given directory as input"
    print "-v, --validate                           Perform validation (LOOCV) when clustering in training mode (default: false)"
    print "-o, --output <output-dir>                Location of training data file (in training mode) or classified images (in classify mode) (default:  <training-data-dir>/" + trainer.TRAINING_DATA_FILENAME + " training mode, <img-dir>_out in classify mode)"
    print "-a, --algorithm <classifier>             Use either \"bf\" (brute force) or \"hist\" (histogram) as the classifier algorithm (default: \"bf\")"
    print "-d, --data <training-data-file>          Use given file as source of training data"
    print "-r, --results <prefix-value>             [USE WITH -c] Check results after classification by inspecting filenames (filename that starts with the given prefix means it should be classified as a positive)"
    
def main(argv):
    if len(argv) == 0:
        show_help()
        sys.exit()

    # Parse parameters.
    training_dir = ""
    output_dir = ""
    query_path = ""
    training_db = ""
    training_mode = False
    classify_mode = False
    classify_mode_alg = classifier.CLASSIFIER_ALG_BF
    results_prefix = ""
    validate = False
    try:
        opts, args = getopt.getopt(argv,"t:vo:c:a:d:r:",["training=","validate=","output=", "classify=", "algorithm=", "data=", "results="])
    except getopt.GetoptError:
        show_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-t", "--training"):
            training_dir = arg
            training_mode = True
        elif opt in ("-o", "--output"):
            output_dir = arg
        elif opt in ("-c", "--classify"):
            query_path = arg
            classify_mode = True
        elif opt in ("-a", "--algorithm"):
            if arg == "bf":
                classify_mode_alg = classifier.CLASSIFIER_ALG_BF
            elif arg == "hist":
                classify_mode_alg = classifier.CLASSIFIER_ALG_HIST
            else:
                print "Illegal value for -a/--algorithm: " + arg
                sys.exit(3)
        elif opt in ("-d", "--data"):
            training_db = arg
        elif opt in ("-r", "--results"):
            results_prefix = arg
        elif opt in ("-v", "--validate"):
            validate = True
            
    if not classify_mode and not training_mode:
        show_help()
        sys.exit(1)
    
    if classify_mode:
        classifier.classify(query_path, training_db, output_dir, results_prefix, classify_mode_alg)
    elif training_mode:
        trainer.train_classifier(training_dir, output_dir, validate)
            
main(sys.argv[1:])
