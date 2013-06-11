package org.prstm;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.Pair;
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

    private final int numTopics;
    private final int numTerms;

    private Configuration conf;

    public TopicModel(Configuration conf, Path... modelPaths) throws IOException {
        this(org.apache.mahout.clustering.lda.cvb.TopicModel.loadModel(conf, modelPaths));

    }

    public TopicModel(Pair<Matrix, Vector> model) {
        this(model.getFirst(), model.getSecond());
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
}
