/**
 * Created by thuy on 31/08/2015.
 */

import java.io._

import scala.collection.mutable
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDAModel, LDA}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
object LDAExample {

  def main(args: Array[String]) {
    val stopwords = scala.io.Source.fromFile("/home/thuy/IdeaProjects/sp/english.stop").getLines().toList
    val conf = new SparkConf().setAppName("LDA").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
   
    //val corpus: RDD[String] = sc.wholeTextFiles("/home/thuy/mini_newsgroups/*").map(_._2)
    val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("/home/thuy/data.csv")
    val corpus: RDD[String] = df.select("content").rdd.map(x => x.getString(0))
    // Split each document into a sequence of terms (words)
    val tokenized: RDD[Seq[String]] =
      corpus.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3).filter(_.forall(java.lang.Character.isLetter)).filter(!stopwords.contains(_)))

    // Choose the vocabulary.
    //   termCounts: Sorted list of (term, termCount) pairs
    val termCounts: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(_._2)
    //   vocabArray: Chosen vocab (removing common terms)
    val numStopwords = 20
    val vocabArray: Array[String] =
      termCounts.takeRight(termCounts.size - numStopwords).map(_._1)
    //   vocab: Map term -> term index
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    // Convert documents into term count vectors
    val documents: RDD[(Long, Vector)] =
      tokenized.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }

    // Set LDA parameters
    val numTopics = 5
    val lda = new LDA()
    val ldaModel = lda.setK(numTopics).setMaxIterations(20).run(documents)

//    val avgLogLikelihood = ldaModel.logLikelihood / documents.count()

    // Print topics, showing top-weighted 10 terms for each topic.



    val writer = new BufferedWriter(new FileWriter("output.txt"))
    try {
      val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
      topicIndices.foreach { case (terms, termWeights) =>
        writer.write("TOPIC:")//writer.write("TOPIC:\n")
        terms.zip(termWeights).foreach { case (term, weight) =>
          writer.write(s"${vocabArray(term.toInt)}\t$weight\n")
        }
        writer.write("\n")
      }
    }
    catch {
      case e:Exception => println(e)
    }
    finally {
      writer.close()
    }

  }
}
