
import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.{functions => fn}

object FPGrowth {

  val spark = SparkSession
    .builder()
    .appName("trans")
    .getOrCreate()


  def read_S3(url: String): DataFrame = {

    var trans_df = spark.read.format("csv")
      .option("sep", "|")
      .option("header", "false")
      .load(url)

    trans_df = trans_df.withColumnRenamed("_c0", "user_id")
    trans_df = trans_df.withColumnRenamed("_c1", "create_date")
    trans_df = trans_df.withColumnRenamed("_c2", "order_week")
    trans_df = trans_df.withColumnRenamed("_c3", "category_list")

    print(trans_df.show())
    return trans_df
  }

  def drop_rep(df: DataFrame): DataFrame = {

    import spark.implicits._

    val category_rdd = df.select("category_list")
      .map(_.getString(0).split(",").distinct)
      .filter(c => (c.length > 1 && c.length <= 10))
    val cate_df = category_rdd.toDF()
    return cate_df
  }

  def FPGrowth(cate_df: DataFrame, support: Double, confidence: Double): DataFrame = {
    val fpgrowth = new FPGrowth().setItemsCol("value").setMinSupport(support).setMinConfidence(confidence)
    val model = fpgrowth.fit(cate_df)
    val freq_df = model.freqItemsets
    val rule_df = model.associationRules
    return rule_df
  }

  def upload_to_S3(df: DataFrame, path: String): Unit = {
    val stringify = fn.udf((vs: Seq[String]) => s"""${vs.mkString("_")}""")
    val df2 = df.withColumn("antecedent", stringify(fn.col("antecedent")))
      .withColumn("consequent", stringify(fn.col("consequent")))

    df2.coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv(path)
  }

  def run_task(url: String): Unit = {

    val dateFormat: SimpleDateFormat = new SimpleDateFormat("YYYY-MM-dd")
    val cal: Calendar = Calendar.getInstance()
    val today = dateFormat.format(cal.getTime())

    val trans_df = read_S3(url)
    val cate_df = drop_rep(trans_df)
    val rule = FPGrowth(cate_df,0.00003,0.3)

    val newest_upload_path = "s3://"
   
    upload_to_S3(rule, newest_upload_path)
  }

  def main(args: Array[String]): Unit = {

    val url = "s3://"

    run_task(url)

  }
}
