/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package trash;

/**
 *
 * @author acastano
 */
import java.io.File;
import java.io.IOException;
//import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
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
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import static org.deeplearning4j.examples.dataExamples.ImagePipelineExample.randNumGen;
import static org.deeplearning4j.examples.feedforward.regression.RegressionMathFunctions.iterations;

/**
 * Reference: architecture partially based on DeepFace: http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf
 * Note: this is a sparse dataset with only 1 example for many of the faces; thus, performance is low.
 * Ideally train on a larger dataset like celebs to get params.
 *
 * Currently set to only use the subset images, names starting with A.
 * Switch to NUM_LABELS & NUM_IMAGES to use full dataset.
 */

public class CNNLFWExample {
//    private static final Logger log = LoggerFactory.getLogger(CNNMnistExample.class);

    public static void main(String[] args) {

        
        
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
            org.nd4j.linalg.dataset.api.iterator.DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 784, outputNum);
            
            int contador = 0;
//            while (dataIter.hasNext()) {
//               DataSet ds =  dataIter.next();
//               System.out.println(ds);
//                contador++;                
//            }
            
            System.out.println("Num Clases: "+dataIter.getLabels().size());
            int seed = 0;
            
            
            
            
            
            
            
            final int numRows = 40;
            final int numColumns = 40;
//            int nChannels = 3;
//            int outputNum = LFWLoader.SUB_NUM_LABELS;
//            int numSamples = LFWLoader.SUB_NUM_IMAGES-4;
            int batchSize =   30;   // numSamples/10;
            int iterations = 5;
            int splitTrainNum = (int) (batchSize*.8);
//            int seed = 123;
            int listenerFreq = iterations/5;
            boolean useSubset = true;
            DataSet lfwNext;
            SplitTestAndTrain trainTest;
            DataSet trainInput;
            List<INDArray> testInput = new ArrayList<>();
            List<INDArray> testLabels = new ArrayList<>();
            
            
            
            
            
            
            
            
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
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                            .name("pool1")
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(3, 3)
                            .name("cnn2")
                            .stride(1,1)
                            .nOut(40)
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                            .name("pool2")
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(3, 3)
                            .name("cnn3")
                            .stride(1,1)
                            .nOut(60)
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                            .name("pool3")
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(2, 2)
                            .name("cnn3")
                            .stride(1,1)
                            .nOut(80)
                            .build())
                    .layer(4, new DenseLayer.Builder()
                            .name("ffn1")
                            .nOut(160)
                            .dropOut(0.5)
                            .build())
                    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(outputNum)
                            .activation("softmax")
                            .build())
                    .backprop(true).pretrain(false);
            new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);
            
            MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
            model.init();
            
//            log.info("Train model....");
            model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
            
            org.nd4j.linalg.dataset.api.iterator.DataSetIterator lfw = dataIter;
            
            while(lfw.hasNext()) {
                lfwNext = lfw.next();
                lfwNext.scale();
                trainTest = lfwNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
                trainInput = trainTest.getTrain(); // get feature matrix and labels for training
                testInput.add(trainTest.getTest().getFeatureMatrix());
                testLabels.add(trainTest.getTest().getLabels());
                model.fit(trainInput);
            }
            
//            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(lfw.getLabels());
            for(int i = 0; i < testInput.size(); i++) {
                INDArray output = model.output(testInput.get(i));
                eval.eval(testLabels.get(i), output);
            }
            INDArray output = model.output(testInput.get(0));
            eval.eval(testLabels.get(0), output);
//            log.info(eval.stats());
//            log.info("****************Example finished********************");
        } catch (IOException ex) {
            java.util.logging.Logger.getLogger(CNNLFWExample.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

}