/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package trash;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 *
 * @author acastano
 */
public class Sample1 {

    public static void main(String args) {
/*
        try {
            // Set path to the labeled images
            String labeledPath = System.getProperty("user.home") + "/lfw";
            
            //create array of strings called labels
            List<String> labels = new ArrayList<>();
            
            //traverse dataset to get each label
            for (File f : new File(labeledPath).listFiles()) {
                labels.add(f.getName());
            }
            
            
            // Instantiating RecordReader. Specify height and width of images.
           // RecordReader recordReader = new ImageRecordReader(28, 28, true, labels);
            
            
            DataSetIterator iter = (DataSetIterator) new RecordReaderDataSetIterator(recordReader, 784, labels.size());
            
            // Point to data path.
           // recordReader.initialize(new FileSplit(new File(labeledPath)));
        } catch (IOException ex) {
            Logger.getLogger(Sample1.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            Logger.getLogger(Sample1.class.getName()).log(Level.SEVERE, null, ex);
        }*/

    }
}
