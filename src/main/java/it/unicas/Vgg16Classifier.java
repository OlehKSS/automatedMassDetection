package it.unicas;

import java.io.File;
import java.io.IOException;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.*;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Hello world!
 *
 */
public class Vgg16Classifier
{
    private static Logger log = LoggerFactory.getLogger(Vgg16Classifier.class);

    private static Model vgg16Model = null;
    private static ComputationGraph vgg16CGraph = null;

    public Vgg16Classifier() throws java.io.IOException
    {
        if (vgg16CGraph == null) {
            try
            {
                loadLocalModelAsGraph();
            }
            catch (java.io.FileNotFoundException)
            {
                downloadModel();
            }

        }
    }

    public void testImg(File imgFile) throws java.io.IOException
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

    public void transferLearning(int numClasses, int seed) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException
    {
        final String featureExtractionLayer = "fc2";
        final int trainPerc = 80;
        final int batchSize = 15;

        log.info(vgg16CGraph.summary());
        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(5e-5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(seed)
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16CGraph)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(numClasses)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0,0.2*(2.0/(4096+numClasses)))) //This weight init dist gave better results than Xavier
                                .activation(Activation.SOFTMAX).build(),
                        featureExtractionLayer)
                .build();
        log.info(vgg16Transfer.summary());
        vgg16CGraph = null;


        //Dataset iterators
        MammogramDataIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = MammogramDataIterator.trainIterator();
        DataSetIterator testIter = MammogramDataIterator.testIterator();

        Evaluation eval;
        eval = vgg16Transfer.evaluate(testIter);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats() + "\n");
        testIter.reset();

        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());
            if (iter % 10 == 0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }

        log.info("Model build complete");
        saveModel(vgg16Transfer);
    }

    private static void loadLocalModel(String modelPath) throws java.io.IOException
    {
        File modelLocation = new File(modelPath);
        // saving of the model
        boolean saveUpdater = true;// True if you want to train your network more in the future

        vgg16Model = ModelSerializer.restoreMultiLayerNetwork(modelLocation, saveUpdater);
    }

    private static void loadLocalModelAsGraph(String modelPath) throws java.io.IOException
    {
        File modelLocation = new File(modelPath);
        // saving of the model
        boolean saveUpdater = true;// True if you want to train your network more in the future

        vgg16CGraph = ModelSerializer.restoreComputationGraph(modelLocation, saveUpdater);
    }

    private static void loadLocalModel() throws java.io.IOException
    {
        String modelPath = "models/vgg16.zip";
        loadLocalModel(modelPath);
    }

    private static void loadLocalModelAsGraph() throws java.io.IOException
    {
        String modelPath = "models/vgg16.zip";
        loadLocalModelAsGraph(modelPath);
    }

    public static void downloadModel() throws java.io.IOException
    {
        ZooModel zooModel = new VGG16();
        log.info("Downloading is started.");
        Model net = zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info("Model is loaded.");

        // saving of the model
        boolean saveUpdater = true;// True if you want to train your network more in the future
        File locationToSave = new File("models/vgg16.zip");
        ModelSerializer.writeModel(net,locationToSave,saveUpdater);
    }

    private static void saveModel(ComputationGraph net) throws java.io.IOException
    {
        boolean saveUpdater = true;// True if you want to train your network more in the future
        File locationToSave = new File("models/vgg16-masses.zip");
        ModelSerializer.writeModel(net,locationToSave,saveUpdater);
    }
}