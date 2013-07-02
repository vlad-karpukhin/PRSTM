package org.rsptm;

import java.io.IOException;
import java.util.Iterator;
import java.util.Random;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author vkarpuhin, created 06.06.13
 */
public class TopicModel implements Configurable, Iterable<MatrixSlice> {
    private static final Logger log = LoggerFactory.getLogger(TopicModel.class);

    private final Matrix topicTermCounts;
    private final Vector topicSums;

    final int numTopics;
    final int numTerms;

    private Configuration conf;

    //private final int numThreads;
    //private Updater[] updaters;

    public TopicModel(Configuration conf, Path... modelPaths) throws IOException {
        this(org.apache.mahout.clustering.lda.cvb.TopicModel.loadModel(conf, modelPaths));

    }

    public TopicModel(Pair<Matrix, Vector> model) {
        this(model.getFirst(), model.getSecond());
    }

    public TopicModel(int numTopics, int numTerms, Random random) {
        this(randomMatrix(numTopics, numTerms, random));
    }

    public TopicModel(Matrix topicTermCounts, Vector topicSums) {
        this.topicTermCounts = topicTermCounts;
        this.topicSums = topicSums;
        this.numTopics = topicSums.size();
        this.numTerms = topicTermCounts.numCols();
    }

    public void setConf(Configuration configuration) {
        this.conf = configuration;
    }

    public Configuration getConf() {
        return conf;
    }

    public Iterator<MatrixSlice> iterator() {
        return null;
    }
    //todo
    private static Pair<Matrix,Vector> randomMatrix(int numTopics, int numTerms, Random random) {
        Matrix topicTermCounts = new DenseMatrix(numTopics, numTerms);
        Vector topicSums = new DenseVector(numTopics);
        if (random != null) {
            for (int x = 0; x < numTopics; x++) {
                for (int term = 0; term < numTerms; term++) {
                    topicTermCounts.viewRow(x).set(term, random.nextDouble());
                }
            }
        }
        for (int x = 0; x < numTopics; x++) {
            topicSums.set(x, random == null ? 1.0 : topicTermCounts.viewRow(x).norm(1));
        }
        return Pair.of(topicTermCounts, topicSums);
    }
}
