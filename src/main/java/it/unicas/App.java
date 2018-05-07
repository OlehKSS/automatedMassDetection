package it.unicas;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_highgui;
//import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_imgcodecs;

import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;

import java.io.File;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Hello world!
 *
 */
public class App 
{
    private static Logger log = LoggerFactory.getLogger(App.class);

    public static void main( String[] args ) throws java.io.IOException
    {
        Vgg16Classifier classifier = new Vgg16Classifier();

        try {
            classifier.transferLearning(2, 12345);
        }
        catch(InvalidKerasConfigurationException|UnsupportedKerasConfigurationException e) {
            log.error("Transfer learning failed.");
            log.error(e.getMessage());
        }
    }

    public static void testOpencv()
    {
        opencv_core.Mat img = opencv_imgcodecs.imread("dataset/images/20586908_6c613a14b80a8591_MG_R_CC_ANON.tif",
                opencv_imgcodecs.IMREAD_GRAYSCALE & opencv_imgcodecs.IMREAD_ANYDEPTH);

        img.convertTo(img, opencv_core.CV_8U);

        opencv_highgui.namedWindow("test", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.imshow("test", img);
        opencv_highgui.waitKey(0);
    }
}
