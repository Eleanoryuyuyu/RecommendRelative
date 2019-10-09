
import java.net.URI

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{functions => fn}

object xgb_lr {


  case class FeaturesClass(
                            pid: Int,
                            catid1: String,
                            catid2: String,
                            catid3: String,
                            list_price: Double,
                            discount: Double,
                            is_cat_male: Double,
                            is_cat_female: Double,
                            imp_1d: Double,
                            imp_uv_1d: Double,
                            click_1d: Double,
                            click_uv_1d: Double,
                            add_1d: Double,
                            add_uv_1d: Double,
                            wishlist_1d: Double,
                            wishlist_uv_1d: Double,
                            in_buyers_1d: Double,
                            in_sales_1d: Double,
                            ctr_1d: Double,
                            acr_1d: Double,
                            page_imp_1d: Double,
                            page_imp_uv_1d: Double,
                            wr_1d: Double,
                            sales_1d: Double,
                            orders_1d: Double,
                            price_1d: Double,
                            gmv_1d: Double,
                            buyers_1d: Double,
                            buyers_male_1d: Double,
                            buyers_female_1d: Double,
                            buyers_neutral_1d: Double,
                            male_rate_1d: Double,
                            female_rate_1d: Double,
                            neutral_rate_1d: Double,
                            refund_1d: Double,
                            refund_rate_1d: Double,
                            repurchase_rate_1d: Double,
                            comment_cnt_1d: Double,
                            score_1d: Double,
                            good_score_rate_1d: Double,
                            imp_7d: Double,
                            imp_uv_7d: Double,
                            click_7d: Double,
                            click_uv_7d: Double,
                            add_7d: Double,
                            add_uv_7d: Double,
                            wishlist_7d: Double,
                            wishlist_uv_7d: Double,
                            in_buyers_7d: Double,
                            in_sales_7d: Double,
                            in_sales: Double,
                            in_buyers: Double,
                            ctr_7d: Double,
                            acr_7d: Double,
                            page_imp_7d: Double,
                            page_imp_uv_7d: Double,
                            page_imp_uv: Double,
                            wr_7d: Double,
                            sales_7d: Double,
                            sales: Double,
                            orders_7d: Double,
                            orders: Double,
                            price_7d: Double,
                            price: Double,
                            gmv_7d: Double,
                            gmv: Double,
                            buyers_7d: Double,
                            buyers_male_7d: Double,
                            buyers_female_7d: Double,
                            buyers_neutral_7d: Double,
                            buyers: Double,
                            male_rate_7d: Double,
                            female_rate_7d: Double,
                            neutral_rate_7d: Double,
                            refund_7d: Double,
                            refund: Double,
                            refund_rate_7d: Double,
                            refund_rate: Double,
                            repurchase_rate_7d: Double,
                            repurchase_rate: Double,
                            comment_cnt_7d: Double,
                            comment_cnt: Double,
                            score_7d: Double,
                            score: Double,
                            score_description: Double,
                            score_quality: Double,
                            good_score_rate_7d: Double,
                            good_score_rate: Double,
                            cid: String,
                            lifecycle_label: Double,
                            lst_visit: Double,
                            lst_add: Double,
                            lst_wishlist: Double,
                            lst_product_buynow: Double,
                            lst_cart_buynow: Double,
                            lst_buynow: Double,
                            coupon_sensitive: Double,
                            is_country_in: Double,
                            is_country_me: Double,
                            is_country_us: Double,
                            is_male: Double,
                            is_female: Double,
                            is_ios: Double,
                            visit_3d_u: Double,
                            visit_7d_u: Double,
                            imp_7d_u: Double,
                            click_7d_u: Double,
                            add_7d_u: Double,
                            wishlist_7d_u: Double,
                            purchase_7d_u: Double,
                            atv_u: Double,
                            real_atv_u: Double,
                            real_pay_u: Double,
                            total_orders_u: Double,
                            refund_cnt_u: Double,
                            comment_cnt_u: Double,
                            cnt: Double,
                            label: Int
                          )

  val spark = SparkSession
    .builder()
    .appName("rank")
    .getOrCreate()

  import spark.implicits._


  val conf = new SparkConf().setAppName("recommend").set("spark.hadoop.validateOutputSpecs", "false").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  val sc = new SparkContext(conf)


  def checkS3PathExist( path: String)  = {

    FileSystem.get(new URI("s3://"), sc.hadoopConfiguration)
      .exists(new Path(path))

  }

  def deleteS3Path( path: String){
    val conf = sc.hadoopConfiguration
    val fs = FileSystem.get(URI.create(path), conf)
    if(checkS3PathExist(path)) fs.delete(new Path(path), true)
  }


  def dataSample(df: DataFrame): DataFrame = {
    val pos_num = df.filter("label=1").count()
    val neg_num = df.filter("label=0").count()
    val ratio = (pos_num * 1.0 / neg_num)
    val pos = df.filter("label=1").sample(false, 0.1)
    val neg = df.filter("label=0").sample(false, 0.1 * ratio)
    val sampled_df = pos.union(neg)
    sampled_df
  }


  def idMapping(onehotPath: String) = {
    import org.json4s.JsonDSL._
    import org.json4s.jackson.JsonMethods._

    val id_info = spark.read.textFile ("s3://")
    val cat1 = id_info.filter(x => x.length() == 3 || x == "-1")
    val cat2 = id_info.filter(x => x.length() == 5 || x == "-1")
    val cat3 = id_info.filter(x => x.length() == 8 || x == "-1")

    val tmp1 = cat1.rdd.zipWithIndex().collect.toMap
    val tmp2 = cat2.rdd.zipWithIndex().collect.toMap
    val tmp3 = cat3.rdd.zipWithIndex().collect.toMap

    val catIdMap = Map("catid1"->tmp1,"catid2"->tmp2,"catid3"->tmp3)
    val jsons = compact(render(catIdMap))

    deleteS3Path(onehotPath)
    val idMap = sc.parallelize(List(jsons)).coalesce(1)
    idMap.saveAsTextFile(onehotPath)

    (tmp1, tmp2, tmp3)

  }


  def oneHot(cat_df: DataFrame, col: String): DataFrame = {


    val encoder = new OneHotEncoder()
      .setInputCol(col)
      .setOutputCol(col + "Vec")
      .setDropLast(false)

    val encoded = encoder.transform(cat_df)

    encoded
  }

  def idProcess(sampled_df: DataFrame, onehotPath: String): DataFrame = {

    val (tmp1, tmp2, tmp3) = idMapping(onehotPath)

    var cat1DF = tmp1.toSeq.toDF("catid1", "catid1Index")
    var cat2DF = tmp2.toSeq.toDF("catid2", "catid2Index")
    var cat3DF = tmp3.toSeq.toDF("catid3", "catid3Index")

    cat1DF = oneHot(cat1DF,"catid1Index")
    cat2DF = oneHot(cat2DF,"catid2Index")
    cat3DF = oneHot(cat3DF,"catid3Index")

    var trans_df = sampled_df

    trans_df = trans_df.join(cat1DF, trans_df("catid1") === cat1DF("catid1"), "inner")
    trans_df = trans_df.join(cat2DF, trans_df("catid2") === cat2DF("catid2"), "inner")
    trans_df = trans_df.join(cat3DF, trans_df("catid3") === cat3DF("catid3"), "inner")

    val colsToRemove =  Array("catid1","catid2","catid3","catid1Index","catid2Index","catid3Index")
    val filteredDF = trans_df.drop(colsToRemove: _*)

    filteredDF

  }

  def xgbInputProcess(df: DataFrame, onehotPath: String): DataFrame = {

    val sampled_df = dataSample(df)
    val new_df = idProcess(sampled_df, onehotPath)
    val filter_col = Array("label", "pid", "cid", "catid1IndexVec", "catid2IndexVec", "catid3IndexVec")
    val feature_col = new_df.columns.diff(filter_col)
    val vectorAssembler = new VectorAssembler()
      .setInputCols(feature_col)
      .setOutputCol("features")
    val assembelData = vectorAssembler.transform(new_df)
    //    val newAssembelData = assembelData.withColumn("features", toDense(fn.col("features")))

    assembelData

  }

  def xgbTrain(trainingDF: DataFrame, xgbModelPath: String): Unit = {

    val xgbParam = Map(
      "eta" -> 0.05f,
      "max_depth" -> 9,
      "subsample" -> 0.8,
      "colsample_bytree" -> 0.8,
      "colsample_bylevel" -> 0.9,
      "eval_metric" -> "auc",
      "objective" -> "binary:logistic",
      "eval_metric" -> "logloss",
      "num_round" -> 100,
      "num_workers" -> 2
    )
    val xgboostModel = new XGBoostClassifier(xgbParam)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .fit(trainingDF)

    val xgboostClassifierModel = xgboostModel.asInstanceOf[XGBoostClassificationModel].setLeafPredictionCol("predLeaf")


    xgboostClassifierModel.write.overwrite().save(xgbModelPath)

  }

  def lrInputProcess(leafDF: DataFrame, idDF: DataFrame): DataFrame = {


    val toVectors = fn.udf((array: Seq[Float]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    val leafDF2 = leafDF.withColumn("predLeaf", toVectors(fn.col("predLeaf")))

    val df_vecIndexer = new VectorIndexer()
      .setInputCol("predLeaf")
      .setOutputCol("predLeafIndex")
      .setMaxCategories(200)
    val leafVec = df_vecIndexer.fit(leafDF2).transform(leafDF2)

    val keys = Array("label", "features")
    val newData = leafDF2.select("label","features","predLeafIndex").join(idDF, keys, "inner")

    val input_col =  Array("predLeafIndex","catid1IndexVec","catid2IndexVec","catid3IndexVec")
    val assembler = new VectorAssembler()
      .setInputCols(input_col)
      .setOutputCol("newFeatures")
    val newInput = assembler.transform(newData)
    newInput.select("label","newFeatures")

  }

  def lrTrain(newTrain: DataFrame, lrModelPath:String): Unit = {

    val logisticRegression = new LogisticRegression()
      .setRegParam(0.3)
      .setMaxIter(100)
      .setLabelCol("label")
      .setFeaturesCol("newFeatures")
      .setPredictionCol("predictionCol")
      .setProbabilityCol("probabilityCol")

    val lrModel = logisticRegression.fit(newTrain)
    val coef = lrModel.coefficients.toArray.mkString("\n")
    val bias = lrModel.intercept
    val lrParam = Array(coef, bias)
    val rddTmp = sc.parallelize(lrParam.toSeq).coalesce(1)

    deleteS3Path(lrModelPath)
    rddTmp.saveAsTextFile(lrModelPath)

  }

  def xgbLRTrain(train: DataFrame, xgbModelPath: String, lrModelPath: String, onehotPath: String): Unit = {

    val trainingAssembel = xgbInputProcess(train, onehotPath)
    val trainingDF = trainingAssembel.select("label", "features").cache()
    val trainingID = trainingAssembel.select("label", "features", "pid", "cid", "catid1Vec", "catid2Vec", "catid3Vec").cache()
    xgbTrain(trainingDF, xgbModelPath)
    val xgboostModel = XGBoostClassificationModel.load(xgbModelPath)
    val trainingLeaf = xgboostModel.transform(trainingDF)
    val newTrain = lrInputProcess(trainingLeaf, trainingID)
    lrTrain(newTrain,lrModelPath)


  }


  def main(args: Array[String]): Unit = {


    val data_path = "s3://"
    val rdd = spark.read.textFile(data_path)
    val header = rdd.first()
    var df = rdd.filter(_ != header).map(_.toString.split("\\|")).map(
      fields => FeaturesClass(
        fields(1).toInt,
        fields(2).toString,
        fields(3).toString,
        fields(4).toString,
        fields(5).toDouble,
        fields(6).toDouble,
        fields(7).toDouble,
        fields(8).toDouble,
        fields(9).toDouble,
        fields(10).toDouble,
        fields(11).toDouble,
        fields(12).toDouble,
        fields(13).toDouble,
        fields(14).toDouble,
        fields(15).toDouble,
        fields(16).toDouble,
        fields(17).toDouble,
        fields(18).toDouble,
        fields(19).toDouble,
        fields(20).toDouble,
        fields(21).toDouble,
        fields(22).toDouble,
        fields(23).toDouble,
        fields(24).toDouble,
        fields(25).toDouble,
        fields(26).toDouble,
        fields(27).toDouble,
        fields(28).toDouble,
        fields(29).toDouble,
        fields(30).toDouble,
        fields(31).toDouble,
        fields(32).toDouble,
        fields(33).toDouble,
        fields(34).toDouble,
        fields(35).toDouble,
        fields(36).toDouble,
        fields(37).toDouble,
        fields(38).toDouble,
        fields(39).toDouble,
        fields(40).toDouble,
        fields(41).toDouble,
        fields(42).toDouble,
        fields(43).toDouble,
        fields(44).toDouble,
        fields(45).toDouble,
        fields(46).toDouble,
        fields(47).toDouble,
        fields(48).toDouble,
        fields(49).toDouble,
        fields(50).toDouble,
        fields(51).toDouble,
        fields(52).toDouble,
        fields(53).toDouble,
        fields(54).toDouble,
        fields(55).toDouble,
        fields(56).toDouble,
        fields(57).toDouble,
        fields(58).toDouble,
        fields(59).toDouble,
        fields(60).toDouble,
        fields(61).toDouble,
        fields(62).toDouble,
        fields(63).toDouble,
        fields(64).toDouble,
        fields(65).toDouble,
        fields(66).toDouble,
        fields(67).toDouble,
        fields(68).toDouble,
        fields(69).toDouble,
        fields(70).toDouble,
        fields(71).toDouble,
        fields(72).toDouble,
        fields(73).toDouble,
        fields(74).toDouble,
        fields(75).toDouble,
        fields(76).toDouble,
        fields(77).toDouble,
        fields(78).toDouble,
        fields(79).toDouble,
        fields(80).toDouble,
        fields(81).toDouble,
        fields(82).toDouble,
        fields(83).toDouble,
        fields(84).toDouble,
        fields(85).toDouble,
        fields(86).toDouble,
        fields(87).toDouble,
        fields(88).toDouble,
        fields(89),
        fields(90).toDouble,
        fields(91).toDouble,
        fields(92).toDouble,
        fields(93).toDouble,
        fields(94).toDouble,
        fields(95).toDouble,
        fields(96).toDouble,
        fields(97).toDouble,
        fields(98).toDouble,
        fields(99).toDouble,
        fields(100).toDouble,
        fields(101).toDouble,
        fields(102).toDouble,
        fields(103).toDouble,
        fields(104).toDouble,
        fields(105).toDouble,
        fields(106).toDouble,
        fields(107).toDouble,
        fields(108).toDouble,
        fields(109).toDouble,
        fields(110).toDouble,
        fields(111).toDouble,
        fields(112).toDouble,
        fields(113).toDouble,
        fields(114).toDouble,
        fields(115).toDouble,
        fields(116).toDouble,
        fields(117).toDouble,
        fields(118).toInt
      )).toDF()


    println(df.show())


    val xgbModelPath = "s3://"
    val lrModelPath = "s3://"
    val onehotPath = "s3://"

    xgbLRTrain(df, xgbModelPath, lrModelPath, onehotPath)

  }
}
