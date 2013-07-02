/*
 * &copy; John Wiley &amp; Sons, Inc
 */
package org.rsptm;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author vkarpuhin, created 14.06.13
 */
public class ModelTrainer {

    private static final Logger LOG = LoggerFactory.getLogger(ModelTrainer.class);

    private final TopicModel model;
    private ThreadPoolExecutor threadPool;
    private BlockingQueue<Runnable> workQueue;
    private final int numTrainThreads;

    public ModelTrainer(TopicModel model, int numTrainThreads) {
        this.model = model;
        this.numTrainThreads = numTrainThreads;

    }

    public void start() {
        LOG.info("Starting training thread-pool with " + numTrainThreads + " threads");
    }

    public void train(Vector document,  int numDocTopicIters) {
        int numTopics = model.numTopics;
        Vector doTopicVector = new DenseVector(new double[numTopics]).assign(1.0/ numTopics);
    }
}
