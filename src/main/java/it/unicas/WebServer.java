package it.unicas;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import javax.servlet.MultipartConfigElement;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import static spark.Spark.*;

/**
 * Web server example
 */
public class WebServer {
    public static void main(String[] args) throws Exception {

        // Set locations for certificates for https encryption
        String keyStoreLocation = "clientkeystore";
        String keyStorePassword = "skymind";
        secure(keyStoreLocation, keyStorePassword, null,null );

        // Load the trained model
        File locationToSave = new File("./models/vgg16.zip");
        ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(locationToSave);


        // make upload directory for user submitted images
        // Images are uploaded, processed and then deleted
        File uploadDir = new File("upload");
        uploadDir.mkdir(); // create the upload directory if it doesn't exist

        // form this string displays an html form to select and upload an image
        String form = "<form method='post' action='getPredictions' enctype='multipart/form-data'>\n" +
                "    <input type='file' name='uploaded_file'>\n" +
                "    <button>Upload picture</button>\n" +
                "</form>";


        // Spark Java configuration to handle requests
        // test request, the url /hello should return "hello world"
        get("/hello", (req, res) -> "Hello World");

        // Request for VGGpredict returns the form to submit an image
        get("VGGpredict", (req, res) -> form);

        // a Post request (note the form uses http post) for
        // getPredictions (note the action attribute or the form)
        // Page prints results and then another form

        post("/getPredictions", (req, res) -> {

            Path tempFile = Files.createTempFile(uploadDir.toPath(), "", "");

            req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));

            try (InputStream input = req.raw().getPart("uploaded_file").getInputStream()) { // getPart needs to use same "name" as input field in form
                Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING);
            }


            // The user submitted file is tempFile, convert to Java File "file"
            File file = tempFile.toFile();

            // Convert file to INDArray
            NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
            INDArray image = loader.asMatrix(file);

            // delete the physical file, if left our drive would fill up over time
            file.delete();

            // Mean subtraction pre-processing step for VGG
            DataNormalization scaler = new VGG16ImagePreProcessor();
            scaler.transform(image);

            //Inference returns array of INDArray, index[0] has the predictions
            INDArray[] output = vgg16.output(false,image);

            // convert 1000 length numeric index of probabilities per label
            // to sorted return top 5 convert to string using helper function VGG16.decodePredictions
            // "predictions" is string of our results
            //String predictions = TrainedModels.VGG16.decodePredictions(output[0]);

            // return results along with form to run another inference
            return "<h4> '" + output  + "' </h4>" +
                    "Would you like to try another" +
                    form;
            //return "<h1>Your image is: '" + tempFile.getName(1).toString() + "' </h1>";

        });
    }
}