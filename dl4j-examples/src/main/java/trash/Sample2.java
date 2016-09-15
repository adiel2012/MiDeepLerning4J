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
import static org.deeplearning4j.examples.dataExamples.ImagePipelineExample.randNumGen;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 *
 * @author acastano
 */
public class Sample2 {

    public static void main(String[] args) {
        try {
            File parentDir = new File("C:\\Users\\acastano\\Downloads\\lfw\\lfw\\lfw");
            String[] allowedExtensions = new String[]{"jpg"};
            FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
            
            
            BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
            InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
            InputSplit trainData = filesInDirSplit[0];
            InputSplit testData = filesInDirSplit[1];
            
            
            ImageRecordReader recordReader = new ImageRecordReader(28, 28, 3, labelMaker);
//        int outputNum = 0;
            
            recordReader.initialize(trainData);
            DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 784, 5749);
            
            int contador = 0;
//            while (dataIter.hasNext()) {
//               DataSet ds =  dataIter.next();
//               System.out.println(ds);
//                contador++;                
//            }
            
            System.out.println("Num Clases: "+dataIter.getLabels().size());
        } catch (IOException ex) {
            Logger.getLogger(Sample2.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
}
