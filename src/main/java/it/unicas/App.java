package it.unicas;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_highgui;
//import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_imgcodecs;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        opencv_core.Mat img = opencv_imgcodecs.imread("dataset/images/20586908_6c613a14b80a8591_MG_R_CC_ANON.tif",
                opencv_imgcodecs.IMREAD_GRAYSCALE & opencv_imgcodecs.IMREAD_ANYDEPTH);

        img.convertTo(img, opencv_core.CV_8U);


        opencv_highgui.namedWindow("test", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.imshow("test", img);
        opencv_highgui.waitKey(0);
        System.out.println( "Hello World!" );
    }
}
