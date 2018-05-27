package org.deeplearning4j.examples.transferlearning.vgg16;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers.MammogramDataIterator;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

/**
 * @author susaneraly on 3/9/17.
 *
 * We use the transfer learning API to construct a new model based of org.deeplearning4j.transferlearning.vgg16
 * We will hold all layers but the very last one frozen and change the number of outputs in the last layer to
 * match our classification task.
 * In other words we go from where fc2 and predictions are vertex names in org.deeplearning4j.transferlearning.vgg16
 *  fc2 -> predictions (1000 classes)
 *  to
 *  fc2 -> predictions (5 classes)
 * The class "FitFromFeaturized" attempts to train this same architecture the difference being the outputs from the last
 * frozen layer is presaved and the fit is carried out on this featurized dataset.
 * When running multiple epochs this can save on computation time.
 */
public class EditLastLayerOthersFrozen {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.class);

    protected static final int numClasses = 2;
    protected static final long seed = 12345;

    private static final int trainPerc = 70;
    private static final int batchSize = 5;
    private static int epochs = 10;
    private static final String featureExtractionLayer = "fc2";
    private static final String SAVE_DIR = new File(System.getProperty("user.home")) + "/deploy/models/";



    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ZooModel zooModel = new VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .updater(new Nesterovs(5e-5))
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(4096).nOut(numClasses)
                    .weightInit(WeightInit.DISTRIBUTION)
                    .dist(new NormalDistribution(0,0.2*(2.0/(4096+numClasses)))) //This weight init dist gave better results than Xavier
                    .activation(Activation.SOFTMAX).build(),
                "fc2")
            .build();
        log.info(vgg16Transfer.summary());


        double max_acc = 0;
        ComputationGraph best_model = vgg16Transfer;

        for(int i = 0; i < epochs; i++) {
            log.info("\n========================= Current epochs: " + (i + 1) + " ====================================");

            //Dataset iterators
            MammogramDataIterator.setup(batchSize, trainPerc);
            DataSetIterator trainIter = MammogramDataIterator.trainIterator();
            DataSetIterator testIter = MammogramDataIterator.testIterator();

            Evaluation eval;
            eval = vgg16Transfer.evaluate(testIter);
            log.info("Eval stats BEFORE fit.....");
            log.info(eval.stats() + "\n");
            testIter.reset();

            int iter = 0;

            while (trainIter.hasNext()) {
                vgg16Transfer.fit(trainIter.next());
                if (iter % 10 == 0) {
                    log.info("Evaluate model at iter " + iter + " ....");
                    eval = vgg16Transfer.evaluate(testIter);
                    log.info(eval.stats());
                    testIter.reset();
                    if (eval.accuracy() > max_acc) {
                        max_acc = eval.accuracy();
                        best_model = vgg16Transfer;
                    }
                }
                iter++;
            }
        }

        log.info("Model build complete");
        File saved_model = new File(SAVE_DIR + "vgg16_mammogram.zip");
        ModelSerializer.writeModel(vgg16Transfer,saved_model,true);
        log.info("Model saved");


    }
}
