/*
 * &copy; John Wiley &amp; Sons, Inc
 */
package org.rsptm;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;

/**
 * @author vkarpuhin, created 18.06.13
 */
public class EmpiricalDistributionEstimates {
    private final Matrix topicTermCounts;
    private final Vector topicSums;

    private final int numTopics;
    private final int numTerms;

    public EmpiricalDistributionEstimates(int numTopics, int numTerms) {
        this.numTopics = numTopics;
        this.numTerms = numTerms;
        topicTermCounts = new SparseMatrix(numTopics, numTerms);
        topicSums = new DenseVector(numTopics);
        for (int x = 0; x < numTopics; x++) {
            for (int term = 0; term < numTerms; term++) {
                topicTermCounts.viewRow(x).set(term, 0);
            }
        }
    }
}
