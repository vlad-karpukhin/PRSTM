/*
 * &copy; John Wiley &amp; Sons, Inc
 */
package org.rsptm;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.BitSet;
import java.util.Iterator;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author vkarpuhin, created 10.06.13
 */
public class SparseVectorsFromCustomFormat extends AbstractJob {
    private static final Logger log = LoggerFactory.getLogger(SparseVectorsFromCustomFormat.class);

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new SparseVectorsFromCustomFormat(), args);
    }

    public int run(String[] args) throws Exception {
        DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
        ArgumentBuilder abuilder = new ArgumentBuilder();
        GroupBuilder gbuilder = new GroupBuilder();

        Option inputDirOpt = DefaultOptionCreator.inputOption().create();

        Option outputDirOpt = DefaultOptionCreator.outputOption().create();

        Option minSupportOpt = obuilder.withLongName("minSupport").withArgument(
            abuilder.withName("minSupport").withMinimum(1).withMaximum(1).create()).withDescription(
            "(Optional) Minimum Support. Default Value: 2").withShortName("s").create();

        /*Option chunkSizeOpt = obuilder.withLongName("chunkSize").withArgument(
            abuilder.withName("chunkSize").withMinimum(1).withMaximum(1).create()).withDescription(
            "The chunkSize in MegaBytes. 100-10000 MB").withShortName("chunk").create();*/

        Option minDFOpt = obuilder.withLongName("minDF").withRequired(false).withArgument(
            abuilder.withName("minDF").withMinimum(1).withMaximum(1).create()).withDescription(
            "The minimum document frequency.  Default is 1").withShortName("md").create();

        Option maxDFPercentOpt = obuilder.withLongName("maxDFPercent").withRequired(false).withArgument(
            abuilder.withName("maxDFPercent").withMinimum(1).withMaximum(1).create()).withDescription(
            "The max percentage of docs for the DF.  Can be used to remove really high frequency terms."
                + " Expressed as an integer between 0 and 100. Default is 99.  If maxDFSigma is also set, it will override this value.")
            .withShortName("x").create();

        Option sequentialAccessVectorOpt = obuilder.withLongName("sequentialAccessVector").withRequired(false)
            .withDescription(
                "(Optional) Whether output vectors should be SequentialAccessVectors. If set true else false")
            .withShortName("seq").create();

        Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
            .create();

        Group group = gbuilder.withName("Options").withOption(minSupportOpt)
            //.withOption(chunkSizeOpt)
            .withOption(outputDirOpt).withOption(inputDirOpt).withOption(minDFOpt)
            .withOption(maxDFPercentOpt)
            .withOption(sequentialAccessVectorOpt)
            .create();

        Parser parser = new Parser();
        parser.setGroup(group);
        parser.setHelpOption(helpOpt);
        CommandLine cmdLine = parser.parse(args);

        if (cmdLine.hasOption(helpOpt)) {
            CommandLineUtil.printHelp(group);
            return -1;
        }

        Path inputFile = new Path((String) cmdLine.getValue(inputDirOpt));
        Path outputFile = new Path((String) cmdLine.getValue(outputDirOpt));


        HadoopUtil.delete(getConf(), outputFile);

        int minSupport = 2;
        if (cmdLine.hasOption(minSupportOpt)) {
            String minSupportString = (String) cmdLine.getValue(minSupportOpt);
            minSupport = Integer.parseInt(minSupportString);
        }

        int minDf = 1;
        if (cmdLine.hasOption(minDFOpt)) {
            minDf = Integer.parseInt(cmdLine.getValue(minDFOpt).toString());
        }
        int maxDFPercent = 99;
        if (cmdLine.hasOption(maxDFPercentOpt)) {
            maxDFPercent = Integer.parseInt(cmdLine.getValue(maxDFPercentOpt).toString());
        }

        boolean sequentialAccessOutput = false;
        if (cmdLine.hasOption(sequentialAccessVectorOpt)) {
            sequentialAccessOutput = true;
        }

        Configuration conf = getConf();
        CorpusStats stats = getCorpusStats(inputFile, minSupport);
        BitSet mask = setTermsToFilter(stats, minDf, maxDFPercent);
        write(stats, mask, outputFile, sequentialAccessOutput);
        return 0;
    }

    CorpusStats getCorpusStats(Path inputPath, int minSupport) throws IOException {
        FileSystem fileSystem = inputPath.getFileSystem(getConf());
        if (!fileSystem.isFile(inputPath)) {
            throw new IllegalArgumentException("inputPath should be a file");
        }
        FSDataInputStream inputStream = fileSystem.open(inputPath);

        BufferedReader bufReader = new BufferedReader(new InputStreamReader(inputStream));
        int docsNum = Integer.valueOf(bufReader.readLine());
        int termsNum = Integer.valueOf(bufReader.readLine());

        int[] termsDF = new int[termsNum];
        RandomAccessSparseVector[] docVectors = new RandomAccessSparseVector[docsNum];

        String line;
        int docCounter = 0;
        while ((line = bufReader.readLine()) != null) {
            // line is docTermsNumber
            int docTermsNum = Integer.valueOf(line);

            String[] termIds = bufReader.readLine().split(" ");
            String[] termFrequencies = bufReader.readLine().split(" ");

            RandomAccessSparseVector docVector = new RandomAccessSparseVector(termsNum, docTermsNum);


            for (int i = 0; i < termFrequencies.length; i++) {
                String frequency = termFrequencies[i];
                if (frequency == null || frequency.isEmpty()) {
                    continue;
                }
                int f = Integer.valueOf(frequency);

                if (f < minSupport) {
                    continue;
                }

                int termId = Integer.valueOf(termIds[i]);

                termsDF[termId]++;
                docVector.setQuick(termId, f);
            }
            docVectors[docCounter] = docVector;
            docCounter++;

        }
        return new CorpusStats(docsNum, termsNum, termsDF, docVectors);
    }

    BitSet setTermsToFilter(CorpusStats stats, int minDF, int maxDFPercent) {
        BitSet mask = new BitSet(stats.termsNum);

        int maxDF = stats.docsNum * maxDFPercent / 100;
        int[] termsDF = stats.termsDF;
        for (int i = 0; i < termsDF.length; i++) {
            int f = termsDF[i];
            if (f < minDF || f > maxDF) {
                mask.set(i);
            }
        }
        return mask;
    }

    void write(CorpusStats stats, BitSet termFilterMask, Path outputFile, boolean sequentialAccessOutput)
        throws IOException {
        FileSystem fileSystem = outputFile.getFileSystem(getConf());
        fileSystem.create(outputFile);
        SequenceFile.Writer writer = SequenceFile
            .createWriter(fileSystem, getConf(), outputFile, LongWritable.class, VectorWritable.class);

        VectorWritable vectorWritable = new VectorWritable();
        LongWritable longWritable = new LongWritable();

        RandomAccessSparseVector[] docVectors = stats.docVectors;
        for (int i = 0; i < docVectors.length; i++) {
            RandomAccessSparseVector docVector = docVectors[i];

            Iterator<Vector.Element> iterator = docVector.iterateNonZero();
            while (iterator.hasNext()){
                Vector.Element element = iterator.next();

                if(termFilterMask.get(element.index())){
                    iterator.remove();
                }
            }

            Vector vectorToWrite = docVector;
            if(sequentialAccessOutput){
                vectorToWrite = new SequentialAccessSparseVector(docVector);
            }
            vectorWritable.set(vectorToWrite);
            longWritable.set(i);
            writer.append(longWritable,vectorWritable);
        }

        writer.sync();

    }

    static class CorpusStats {
        final int docsNum;
        final int termsNum;
        final int[] termsDF;
        final RandomAccessSparseVector[] docVectors;

        CorpusStats(int docsNum, int termsNum, int[] termsDF, RandomAccessSparseVector[] docVectors) {
            this.docsNum = docsNum;
            this.termsNum = termsNum;
            this.termsDF = termsDF;
            this.docVectors = docVectors;
        }
    }
}
