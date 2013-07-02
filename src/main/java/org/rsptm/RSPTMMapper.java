/*
 * &copy; John Wiley &amp; Sons, Inc
 */
package org.rsptm;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author vkarpuhin, created 14.06.13
 */
public class RSPTMMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

    private static final Logger LOG = LoggerFactory.getLogger(RSPTMMapper.class);

    private ModelTrainer modelTrainer;
    private int maxIters;
    private int numTopics;

    protected ModelTrainer getModelTrainer() {
        return modelTrainer;
    }

    protected int getMaxIters() {
        return maxIters;
    }

    protected int getNumTopics() {
        return numTopics;
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        LOG.info("Retrieving configuration");
        Configuration conf = context.getConfiguration();
        long seed = conf.getLong(RSPTMDriver.RANDOM_SEED, 1234L);
        numTopics = conf.getInt(RSPTMDriver.NUM_TOPICS, -1);
        int numTerms = conf.getInt(RSPTMDriver.NUM_TERMS, -1);
        maxIters = conf.getInt(RSPTMDriver.MAX_ITERATIONS_PER_DOC, 10);
        int numTrainThreads = conf.getInt(RSPTMDriver.NUM_TRAIN_THREADS, 4);

        LOG.info("Initializing read model");
        TopicModel model;

        Path[] modelPaths = RSPTMDriver.getModelPaths(conf);
        if (modelPaths != null && modelPaths.length > 0) {
            model = new TopicModel(conf, modelPaths);
        } else {
            LOG.info("No model files found, initializing with random values");
            model = new TopicModel(numTopics, numTerms, RandomUtils.getRandom(seed));
        }




        LOG.info("Initializing model trainer");
        modelTrainer = new ModelTrainer(model, numTrainThreads);
        modelTrainer.start();

    }

    @Override
    public void map(IntWritable docId, VectorWritable document, Context context)
        throws IOException, InterruptedException{

        modelTrainer.train(document.get(), maxIters);
    }


}
