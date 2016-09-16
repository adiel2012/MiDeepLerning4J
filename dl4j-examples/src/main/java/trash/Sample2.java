/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package trash;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ScaleImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import static org.deeplearning4j.examples.dataExamples.ImagePipelineExample.randNumGen;
import static org.deeplearning4j.examples.feedforward.regression.RegressionMathFunctions.iterations;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author acastano
 */
public class Sample2 {

    public static void main(String[] args) {
        
        //  http://www.cs.toronto.edu/%7Ehinton/absps/guideTR.pdf 
        try {
            int nChannels = 3;
            
            File parentDir = new File("C:\\Users\\acastano\\Downloads\\lfw\\lfw\\lfw");
            String[] allowedExtensions = new String[]{"jpg"};
            FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
            
            
            BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
            InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
            InputSplit trainData = filesInDirSplit[0];
            InputSplit testData = filesInDirSplit[1];
            
            
            ImageRecordReader recordReader = new ImageRecordReader(28, 28, nChannels, labelMaker);
        int outputNum = 5749;
            
            recordReader.initialize(trainData);
            DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 784, outputNum);
            
            int contador = 0;
//            while (dataIter.hasNext()) {
//               DataSet ds =  dataIter.next();
//               System.out.println(ds);
//                contador++;                
//            }
            
            System.out.println("Num Clases: "+dataIter.getLabels().size());
            int seed = 0;
            
            
            
            
            
        
            
            
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel
        new ConvolutionLayerSetup(builder,28,28,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


//        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
            int nEpochs = 2;
            
            Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(dataIter);
//            log.info("*** Completed epoch {} ***", i);
//
//            log.info("Evaluate model....");
//            Evaluation eval = new Evaluation(outputNum);
//            while(mnistTest.hasNext()){
//                DataSet ds = mnistTest.next();
//                INDArray output = model.output(ds.getFeatureMatrix(), false);
//                eval.eval(ds.getLabels(), output);
//            }
//            log.info(eval.stats());
//            mnistTest.reset();
        }
//        log.info("****************Example finished********************");

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        } catch (IOException ex) {
            Logger.getLogger(Sample2.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
}
