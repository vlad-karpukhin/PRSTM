/*
 * &copy; John Wiley &amp; Sons, Inc
 */
package org.prstm;

import java.util.Iterator;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author vkarpuhin, created 06.06.13
 */
public class TopicModel implements Configurable, Iterable<MatrixSlice> {
    private static final Logger log = LoggerFactory.getLogger(TopicModel.class);

    //private final Matrix topicTermCounts;
    //private final int numTopics;
    //private final int numTerms;

    private Configuration conf;

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
