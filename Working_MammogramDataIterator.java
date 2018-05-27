package org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Automatically loads the dataset from the masses folder
 * Modified from FlowerDataSetIterator
 * @author susaneraly on 3/9/17.
 */
public class MammogramDataIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MammogramDataIterator.class);
    private static final Random rng  = new Random(13);
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 2;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize;
    public static final String[] allowedExtensions = {"bmp", "gif", "jpg", "jpeg", "jp2", "pbm", "pgm", "ppm", "pnm",
        "png", "tif", "tiff", "exr", "webp", "BMP", "GIF", "JPG", "JPEG", "JP2", "PBM", "PGM", "PPM", "PNM",
        "PNG", "TIF", "TIFF", "EXR", "WEBP"};
    private static final String MAMMO_TRAIN_DIR = "C:/JAVA_dataset/DataSet 24-05/Train01";
    private static final String MAMMO_TEST_DIR = "C:/JAVA_dataset/DataSet 24-05/Test";

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);

    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData);

    }

    public static void setup(int batchSizeArg, int trainPerc) throws IOException {
        batchSize = batchSizeArg;

        //File MAMMO_TRAIN_DIR = new File(System.getProperty("user.home"), "/deploy/train_set01");
        //File MAMMO_TEST_DIR = new File(System.getProperty("user.home"), "/deploy/test_set");

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File parentDir = new File(MAMMO_TRAIN_DIR);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 100);
        trainData = filesInDirSplit[0];

        parentDir = new File(MAMMO_TEST_DIR);
        filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }
        filesInDirSplit = filesInDir.sample(pathFilter, 100);
        testData = filesInDirSplit[0];
    }

    private static DataSetIterator makeIterator(InputSplit split) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(split);
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        iter.setPreProcessor( new VGG16ImagePreProcessor());
        return iter;
    }
}
