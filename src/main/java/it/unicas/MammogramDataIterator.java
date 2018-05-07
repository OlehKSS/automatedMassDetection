package it.unicas;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ArchiveUtils;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Random;

/**
 * Automatically loads the dataset from the masses folder
 * Modified from FlowerDataSetIterator
 * @author susaneraly on 3/9/17.
 */
public class MammogramDataIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MammogramDataIterator.class);

    private static final String DATA_DIR = "/masses";

    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random rng  = new Random(13);

    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 2;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize;

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);

    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData);

    }

    public static void setup(int batchSizeArg, int trainPerc) throws IOException {
        batchSize = batchSizeArg;
        File parentDir = new File(DATA_DIR);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

    private static DataSetIterator makeIterator(InputSplit split) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(split);
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        iter.setPreProcessor( new VGG16ImagePreProcessor());
        return iter;
    }

}
