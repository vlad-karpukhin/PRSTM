package org.rsptm;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;

/**
 * @author vkarpuhin, created 06.06.13
 */
public class RSPTMDriver extends AbstractJob {


    public static final String NUM_TOPICS = "num_topics";
    public static final String NUM_TERMS = "num_terms";
    public static final String DOC_TOPIC_SMOOTHING = "doc_topic_smoothing";
    public static final String TERM_TOPIC_SMOOTHING = "term_topic_smoothing";
    public static final String DICTIONARY = "dictionary";
    public static final String DOC_TOPIC_OUTPUT = "doc_topic_output";
    public static final String MODEL_TEMP_DIR = "topic_model_temp_dir";
    public static final String ITERATION_BLOCK_SIZE = "iteration_block_size";
    public static final String RANDOM_SEED = "random_seed";
    public static final String TEST_SET_FRACTION = "test_set_fraction";
    public static final String NUM_TRAIN_THREADS = "num_train_threads";
    public static final String NUM_UPDATE_THREADS = "num_update_threads";
    public static final String MAX_ITERATIONS_PER_DOC = "max_doc_topic_iters";
    public static final String MODEL_WEIGHT = "prev_iter_mult";
    public static final String NUM_REDUCE_TASKS = "num_reduce_tasks";
    public static final String BACKFILL_PERPLEXITY = "backfill_perplexity";
    private static final String MODEL_PATHS = "mahout.lda.cvb.modelPath";

    private static final Logger LOG = LoggerFactory.getLogger(RSPTMDriver.class);

    public int run(String[] args) throws Exception {
        addInputOption();
        addOutputOption();
        addOption(DefaultOptionCreator.maxIterationsOption().create());
        addOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION, "cd", "The convergence delta value", "0");
        addOption(DefaultOptionCreator.overwriteOption().create());

        addOption(NUM_TOPICS, "k", "Number of topics to learn", true);
        addOption(NUM_TERMS, "nt", "Vocabulary size", true);

        addOption(DOC_TOPIC_OUTPUT, "dt", "Output path for the training doc/topic distribution",
            false);
        addOption(MODEL_TEMP_DIR, "mt", "Path to intermediate model path (useful for restarting)",
            false);

        addOption(ITERATION_BLOCK_SIZE, "block", "Number of iterations per perplexity check", "10");

        addOption(MAX_ITERATIONS_PER_DOC, "mipd",
            "max number of iterations per doc for p(topic|doc) learning", "10");
        addOption(NUM_REDUCE_TASKS, null,
            "number of reducers to use during model estimation", "10");

        if (parseArguments(args) == null) {
            return -1;
        }

        int numTopics = Integer.parseInt(getOption(NUM_TOPICS));
        Path inputPath = getInputPath();
        Path topicModelOutputPath = getOutputPath();
        int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
        int iterationBlockSize = Integer.parseInt(getOption(ITERATION_BLOCK_SIZE));
        double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
        int maxItersPerDoc = Integer.parseInt(getOption(MAX_ITERATIONS_PER_DOC));
        int numTerms = Integer.parseInt(getOption(NUM_TERMS));
        Path docTopicOutputPath = hasOption(DOC_TOPIC_OUTPUT) ? new Path(getOption(DOC_TOPIC_OUTPUT)) : null;

        Path modelTempPath = hasOption(MODEL_TEMP_DIR)
            ? new Path(getOption(MODEL_TEMP_DIR))
            : getTempPath("topicModelState");

        long seed = hasOption(RANDOM_SEED)
            ? Long.parseLong(getOption(RANDOM_SEED))
            : System.nanoTime() % 10000;

        int numReduceTasks = Integer.parseInt(getOption(NUM_REDUCE_TASKS));


        return 0;
    }

    public static int run(Configuration conf,
                          Path inputPath,
                          Path topicModelOutputPath,
                          int numTopics,
                          int numTerms,
                          int maxIterations,
                          int iterationBlockSize,
                          double convergenceDelta,
                          Path docTopicOutputPath,
                          Path topicModelStateTempPath,
                          long randomSeed,
                          int maxItersPerDoc,
                          int numReduceTasks,
                          boolean backfillPerplexity)
        throws ClassNotFoundException, IOException, InterruptedException {

        // todo
        /*
        String infoString = "Will run Collapsed Variational Bayes (0th-derivative approximation) "
            + "learning for LDA on {} (numTerms: {}), finding {}-topics, with document/topic prior {}, "
            + "topic/term prior {}.  Maximum iterations to run will be {}, unless the change in "
            + "perplexity is less than {}.  Topic model output (p(term|topic) for each topic) will be "
            + "stored {}.  Random initialization seed is {}, holding out {} of the data for perplexity "
            + "check\n";

            log.info(infoString, new Object[] {inputPath, numTerms, numTopics, alpha, eta, maxIterations,
        convergenceDelta, topicModelOutputPath, randomSeed, testFraction});
    infoString = dictionaryPath == null
               ? "" : "Dictionary to be used located " + dictionaryPath.toString() + '\n';
    infoString += docTopicOutputPath == null
               ? "" : "p(topic|docId) will be stored " + docTopicOutputPath.toString() + '\n';
    log.info(infoString);

            */


        FileSystem fs = FileSystem.get(topicModelStateTempPath.toUri(), conf);
        int iterationNumber = getCurrentIterationNumber(conf, topicModelStateTempPath, maxIterations);
        LOG.info("Current iteration number: {}", iterationNumber);

        conf.set(NUM_TOPICS, String.valueOf(numTopics));
        conf.set(NUM_TERMS, String.valueOf(numTerms));
        //conf.set(DOC_TOPIC_SMOOTHING, String.valueOf(alpha));
        //conf.set(TERM_TOPIC_SMOOTHING, String.valueOf(eta));
        conf.set(RANDOM_SEED, String.valueOf(randomSeed));
        //conf.set(NUM_TRAIN_THREADS, String.valueOf(numTrainThreads));
        //conf.set(NUM_UPDATE_THREADS, String.valueOf(numUpdateThreads));
        conf.set(MAX_ITERATIONS_PER_DOC, String.valueOf(maxItersPerDoc));

        //conf.set(TEST_SET_FRACTION, String.valueOf(testFraction));

        long startTime = System.currentTimeMillis();
        while (iterationNumber < maxIterations) {


            // update model
            iterationNumber++;
            LOG.info("About to run iteration {} of {}", iterationNumber, maxIterations);
            Path modelInputPath = modelPath(topicModelStateTempPath, iterationNumber - 1);
            Path modelOutputPath = modelPath(topicModelStateTempPath, iterationNumber);
            runIteration(conf, inputPath, modelInputPath, modelOutputPath, iterationNumber,
                maxIterations, numReduceTasks);



        }
        LOG.info("Completed {} iterations in {} seconds", iterationNumber,
            (System.currentTimeMillis() - startTime)/1000);
        //LOG.info("Perplexities: ({})", Joiner.on(", ").join(perplexities));




        return 0;
    }

    public static void runIteration(Configuration conf, Path corpusInput, Path modelInput, Path modelOutput,
                                    int iterationNumber, int maxIterations, int numReduceTasks) throws IOException, ClassNotFoundException, InterruptedException {

        String jobName = String.format("Iteration %d of %d, input path: %s",
            iterationNumber, maxIterations, modelInput);
        LOG.info("About to run: " + jobName);
        Job job = new Job(conf, jobName);

        job.setMapperClass(RSPTMMapper.class);
        job.setNumReduceTasks(numReduceTasks);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(VectorWritable.class);
        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        FileInputFormat.addInputPath(job, corpusInput);
        FileOutputFormat.setOutputPath(job, modelOutput);

        setModelPaths(job, modelInput);
        HadoopUtil.delete(conf, modelOutput);
        if (!job.waitForCompletion(true)) {
            throw new InterruptedException(String.format("Failed to complete iteration %d stage 1",
                iterationNumber));
        }

    }

    private static void setModelPaths(Job job, Path modelPath) throws IOException {
        Configuration conf = job.getConfiguration();
        if (modelPath == null || !FileSystem.get(modelPath.toUri(), conf).exists(modelPath)) {
            return;
        }
        FileStatus[] statuses = FileSystem.get(modelPath.toUri(), conf).listStatus(modelPath, PathFilters.partFilter());
        Preconditions.checkState(statuses.length > 0, "No part files found in model path '%s'", modelPath.toString());
        String[] modelPaths = new String[statuses.length];
        for (int i = 0; i < statuses.length; i++) {
            modelPaths[i] = statuses[i].getPath().toUri().toString();
        }
        conf.setStrings(MODEL_PATHS, modelPaths);
    }

    public static Path[] getModelPaths(Configuration conf) {
        String[] modelPathNames = conf.getStrings(MODEL_PATHS);
        if (modelPathNames == null || modelPathNames.length == 0) {
            return null;
        }
        Path[] modelPaths = new Path[modelPathNames.length];
        for (int i = 0; i < modelPathNames.length; i++) {
            modelPaths[i] = new Path(modelPathNames[i]);
        }
        return modelPaths;
    }

    private static int getCurrentIterationNumber(Configuration config, Path modelTempDir, int maxIterations)
        throws IOException {
        FileSystem fs = FileSystem.get(modelTempDir.toUri(), config);
        int iterationNumber = 1;
        Path iterationPath = modelPath(modelTempDir, iterationNumber);
        while (fs.exists(iterationPath) && iterationNumber <= maxIterations) {
            LOG.info("Found previous state: " + iterationPath);
            iterationNumber++;
            iterationPath = modelPath(modelTempDir, iterationNumber);
        }
        return iterationNumber - 1;
    }

    public static Path modelPath(Path topicModelStateTempPath, int iterationNumber) {
        return new Path(topicModelStateTempPath, "model-" + iterationNumber);
    }
}
