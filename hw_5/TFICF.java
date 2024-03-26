import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import scala.Tuple2;

import java.util.*;

/*
 * Main class of the TFICF Spark implementation.
 * Author: Tyler Stocksdale
 * Date:   10/31/2017
 */
public class TFICF {

        static boolean DEBUG = false;

    public static void main(String[] args) throws Exception {
        // Check for correct usage
        if (args.length != 1) {
            System.err.println("Usage: TFICF <input dir>");
            System.exit(1);
        }

                // Create a Java Spark Context
                SparkConf conf = new SparkConf().setAppName("TFICF");
                JavaSparkContext sc = new JavaSparkContext(conf);

                // Load our input data
                // Output is: ( filePath , fileContents ) for each file in inputPath
                String inputPath = args[0];
                JavaPairRDD<String,String> filesRDD = sc.wholeTextFiles(inputPath);

                // Get/set the number of documents (to be used in the ICF job)
                long numDocs = filesRDD.count();

                //Print filesRDD contents
                if (DEBUG) {
                        List<Tuple2<String, String>> list = filesRDD.collect();
                        System.out.println("------Contents of filesRDD------");
                        for (Tuple2<String, String> tuple : list) {
                                System.out.println("(" + tuple._1 + ") , (" + tuple._2.trim() + ")");
                        }
                        System.out.println("--------------------------------");
                }

                /* 
                 * Initial Job
                 * Creates initial JavaPairRDD from filesRDD
                 * Contains each word@document from the corpus and also attaches the document size for 
                 * later use
                 * 
                 * Input:  ( filePath , fileContents )
                 * Map:    ( (word@document) , docSize )
                 */
                JavaPairRDD<String,Integer> wordsRDD = filesRDD.flatMapToPair(
                        new PairFlatMapFunction<Tuple2<String,String>,String,Integer>() {
                                @Override public Iterator<Tuple2<String,Integer>> call(Tuple2<String,String> x) {
                                        // Collect data attributes
                                        String[] filePath = x._1.split("/");
                                        String document = filePath[filePath.length-1];
                                        String fileContents = x._2;
                                        String[] words = fileContents.split("\\s+");
                                        int docSize = words.length;

                                        // Output to Arraylist
                                        ArrayList<Tuple2<String, Integer>> ret = new ArrayList<>();
                                        for (String word : words) {
                                                ret.add(new Tuple2<>(word.trim() + "@" + document, docSize));
                                        }
                                        return ret.iterator();
                                }
                        }
                );

                //Print wordsRDD contents
                if (DEBUG) {
                        List<Tuple2<String, Integer>> list = wordsRDD.collect();
                        System.out.println("------Contents of wordsRDD------");
                        for (Tuple2<String, Integer> tuple : list) {
                                System.out.println("(" + tuple._1 + ") , (" + tuple._2 + ")");
                        }
                        System.out.println("--------------------------------");
                }

                /* 
                 * TF Job (Word Count Job + Document Size Job)
                 * Gathers all data needed for TF calculation from wordsRDD
                 *
                 * Input:  ( (word@document) , docSize )
                 * Map:    ( (word@document) , (1/docSize) )
                 * Reduce: ( (word@document) , (wordCount/docSize) )
                 */
                JavaPairRDD<String, String> tfMappedRDD = wordsRDD.flatMapToPair(
                                new PairFlatMapFunction<Tuple2<String, Integer>, String, String>() {
                                        @Override
                                        public Iterator<Tuple2<String, String>> call(Tuple2<String, Integer> tuple) throws Exception {
                                                String[] parts = tuple._1.split("@"); // Split the  word@document
                                                String word = parts[0];
                                                String document = parts[1];
                                                int docSize = tuple._2;
                                                List<Tuple2<String, String>> res = new ArrayList<>();
                                                res.add(new Tuple2<String, String>(word + "@" + document, "1/" + docSize));
                                                return res.iterator();
                                        }
                                }
                                );

                // Reduce by key to aggregate counts
                JavaPairRDD<String, String> tfRDD = tfMappedRDD.reduceByKey(
                                new Function2<String, String, String>() {
                                        @Override
                                        public String call(String x, String y) throws Exception {
                                                String[] a = x.split("/");
                                                String[] b = y.split("/");
                                                int wordCount = Integer.parseInt(a[0]) + Integer.parseInt(b[0]);
                                                return wordCount + "/" + a[1];
                                        }
                                }
                                );
                //Print tfRDD contents
                if (DEBUG) {
                        List<Tuple2<String, String>> list = tfRDD.collect();
                        System.out.println("-------Contents of tfRDD--------");
                        for (Tuple2<String, String> tuple : list) {
                                System.out.println("(" + tuple._1 + ") , (" + tuple._2 + ")");
                        }
                        System.out.println("--------------------------------");
                }

                /*
                 * ICF Job
                 * Gathers all data needed for ICF calculation from tfRDD
                 *
                 * Input:  ( (word@document) , (wordCount/docSize) )
                 * Map:    ( word , (1/document) )
                 * Reduce: ( word , (numDocsWithWord/document1,document2...) )
                 * Map:    ( (word@document) , (numDocs/numDocsWithWord) )
                 */
                                //Print icfRDD contents
                JavaPairRDD<String, String> icfMappedRDD = tfRDD.flatMapToPair(
                                new PairFlatMapFunction<Tuple2<String, String>, String, String>() {
                                        @Override
                                        public Iterator<Tuple2<String, String>> call(Tuple2<String, String> tuple) throws Exception{
                                                String[] parts = tuple._1.split("@"); // Splitting word@document
                                                String word = parts[0];
                                                String document = parts[1];
                                                List<Tuple2<String, String>> result = new ArrayList<>();
                                                result.add(new Tuple2<String, String>(word, "1/" + document));
                                                return result.iterator();
                                        }
                                }
                                );
                JavaPairRDD<String, String> icfReducedRDD = icfMappedRDD.reduceByKey(
                                new Function2<String, String, String>() {
                                        @Override
                                        public String call(String x, String y) throws Exception{
                                                String[] docA = x.split("/");
                                                String[] docB = y.split("/");
                                                int numDocsWithWord = Integer.parseInt(docA[0]) + Integer.parseInt(docB[0]);
                                                StringBuilder documents = new StringBuilder(docA[1]);
                                                for (int i = 2; i < docA.length; i++) {
                                                        documents.append(",").append(docA[i]);
                                                }
                                                for (int i = 1; i < docB.length; i++) {
                                                        documents.append(",").append(docB[i]);
                                                }
                                                return numDocsWithWord + "/" + documents.toString();
                                        }
                                }
                );

                JavaPairRDD<String, String> icfRDD = icfReducedRDD.flatMapToPair(
                                new PairFlatMapFunction<Tuple2<String, String>, String, String>() {
                                        @Override
                                        public Iterator<Tuple2<String, String>> call(Tuple2<String, String> tuple) throws Exception{
                                                String word = tuple._1;
                                                String[] parts = tuple._2.split("/");
                                                int numDocsWithWord = Integer.parseInt(parts[0]);
                                                String[] documents = parts[1].split(","); 
                                                List<Tuple2<String, String>> res = new ArrayList<>();
                                                for (String document : documents) {
                                                        res.add(new Tuple2<String, String>(word + "@" + document, numDocs + "/" + numDocsWithWord));
                                                }
                                                return res.iterator();
                                        }
                                }
                                );

                if (DEBUG) {
                        List<Tuple2<String, String>> list = icfRDD.collect();
                        System.out.println("-------Contents of icfRDD-------");
                        for (Tuple2<String, String> tuple : list) {
                                System.out.println("(" + tuple._1 + ") , (" + tuple._2 + ")");
                        }
                        System.out.println("--------------------------------");
                }

                /*
                 * TF * ICF Job
                 * Calculates final TFICF value from tfRDD and icfRDD
                 *
                 * Input:  ( (word@document) , (wordCount/docSize) )          [from tfRDD]
                 * Map:    ( (word@document) , TF )
                 * 
                 * Input:  ( (word@document) , (numDocs/numDocsWithWord) )    [from icfRDD]
                 * Map:    ( (word@document) , ICF )
                 * 
                 * Union:  ( (word@document) , TF )  U  ( (word@document) , ICF )
                 * Reduce: ( (word@document) , TFICF )
                 * Map:    ( (document@word) , TFICF )
                 *
                 * where TF    = log( wordCount/docSize + 1 )
                 * where ICF   = log( (Total numDocs in the corpus + 1) / (numDocsWithWord in the corpus + 1) )
                 * where TFICF = TF * ICF
                 */
                JavaPairRDD<String,Double> tfFinalRDD = tfRDD.mapToPair(
                        new PairFunction<Tuple2<String,String>,String,Double>() {
                                public Tuple2<String,Double> call(Tuple2<String,String> x){
                                        double wordCount = Double.parseDouble(x._2.split("/")[0]);
                                        double docSize = Double.parseDouble(x._2.split("/")[1]);
                                        double TF = wordCount/docSize;
                                        return new Tuple2<String, Double>(x._1, TF);
                                }
                        }
                );
                JavaPairRDD<String,Double> idfFinalRDD = icfRDD./**MAP**/mapToPair(
                         new PairFunction<Tuple2<String,String>,String,Double>() {
								@Override
                                public Tuple2<String,Double> call(Tuple2<String,String> tuple) throws Exception{
                                        String[] parts = tuple._2.split("/");
                                        double numdocs = Double.parseDouble(parts[0]);
                                        double numDocsWithWord = Double.parseDouble(parts[1]);
                                        double ICF = Math.log10((numdocs+1)/(numDocsWithWord+1));
                                        return new Tuple2<String, Double>(tuple._1, ICF);
                                }
                        }
                );
                JavaPairRDD<String,Double> tficfRDD = tfFinalRDD.union(idfFinalRDD)./**REDUCE**/reduceByKey(
                        new Function2<Double,Double,Double>() {
								@Override
                                public Double call(Double TF,Double ICF) throws Exception{
                                                double TFICF = Math.log10(TF+1)*ICF;
                                                return TFICF;
                                }
                        }
                )./**MAP**/flatMapToPair(
                        new PairFlatMapFunction<Tuple2<String,Double>,String,Double>() {
								@Override
                                public Iterator<Tuple2<String,Double>> call(Tuple2<String,Double> tuple) throws Exception{
                                        String[] parts = tuple._1.split("@");
                                        ArrayList<Tuple2<String,Double>> res = new ArrayList<>();
                                        res.add(new Tuple2<>(parts[1] + "@" + parts[0], tuple._2));
                                        return res.iterator();
                                }
                        }

                );

                //Print tficfRDD contents in sorted order
                Map<String, Double> sortedMap = new TreeMap<>();
                List<Tuple2<String, Double>> list = tficfRDD.collect();
                for (Tuple2<String, Double> tuple : list) {
                        sortedMap.put(tuple._1, tuple._2);
                }
                if(DEBUG) System.out.println("-------Contents of tficfRDD-------");
                for (String key : sortedMap.keySet()) {
                        System.out.println(key + "\t" + sortedMap.get(key));
                }
                if(DEBUG) System.out.println("--------------------------------");
        }
}


