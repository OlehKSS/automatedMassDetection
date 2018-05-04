package it.unicas;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_highgui;
//import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.io.File;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.*;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;


/**
 * Hello world!
 *
 */
public class App 
{
    static Model vgg16Model;
    static ComputationGraph vgg16CGraph;
    public static void main( String[] args ) throws java.io.IOException
    {
        loadLocalModelAsGraph();

        File cat = new File("dataset/pets/cat.jpg");
        File dog = new File("dataset/pets/puppy-dog.jpg");
        File elephant = new File("dataset/pets/elephant.jpg");

        System.out.println("Cat image");
        testImg(cat);


        System.out.println("Dog image");
        testImg(dog);

        System.out.println("Elephant image");
        testImg(elephant);
    }

    public static void testImg(File imgFile) throws java.io.IOException
    {
        NativeImageLoader loader = new NativeImageLoader(224, 224,3);
        INDArray img = loader.asMatrix(imgFile);
        // Mean subtraction pre-processing step for VGG
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(img);
        // actual prediction
        //INDArray[] output = vgg16.output(false,image);
        INDArray[] output = vgg16CGraph.output(false, img);
        // convert 1000 length numeric index of probabilities per label
        // to sorted return top 5 convert to string using helper function VGG16.decodePredictions
        // "predictions" is string of our results
        String predictions = TrainedModels.VGG16.decodePredictions(output[0]);
        System.out.println(predictions);
    }

    public static void loadLocalModel() throws java.io.IOException
    {
        File modelLocation = new File("models/vgg16.zip");
        // saving of the model
        boolean saveUpdater = true;// True if you want to train your network more in the future

        vgg16Model = ModelSerializer.restoreMultiLayerNetwork(modelLocation, saveUpdater);
    }

    public static void loadLocalModelAsGraph() throws java.io.IOException
    {
        File modelLocation = new File("models/vgg16.zip");
        // saving of the model
        boolean saveUpdater = true;// True if you want to train your network more in the future

        vgg16CGraph = ModelSerializer.restoreComputationGraph(modelLocation, saveUpdater);
    }

    public static void downloadModel() throws java.io.IOException
    {
        ZooModel zooModel = new VGG16();
        System.out.println("Downloading is started.");
        Model net = zooModel.initPretrained(PretrainedType.IMAGENET);
        System.out.println("Model is loaded.");

        // saving of the model
        boolean saveUpdater = true;// True if you want to train your network more in the future
        File locationToSave = new File("models/vgg16.zip");
        ModelSerializer.writeModel(net,locationToSave,saveUpdater);
    }

    public static void testOpencv()
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
