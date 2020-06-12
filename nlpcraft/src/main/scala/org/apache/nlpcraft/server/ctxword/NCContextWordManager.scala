/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.nlpcraft.server.ctxword

import com.google.common.util.concurrent.AtomicDouble
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import io.opencensus.trace.Span
import org.apache.http.HttpResponse
import org.apache.http.client.ResponseHandler
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.HttpClients
import org.apache.http.util.EntityUtils
import org.apache.ignite.IgniteCache
import org.apache.nlpcraft.common.ascii.NCAsciiTable
import org.apache.nlpcraft.common.config.NCConfigurable
import org.apache.nlpcraft.common.nlp.core.NCNlpCoreManager
import org.apache.nlpcraft.common.{NCE, NCService}
import org.apache.nlpcraft.server.ignite.NCIgniteInstance
import org.apache.nlpcraft.server.mdo.NCContextWordConfigMdo
import org.apache.nlpcraft.server.nlp.core.{NCNlpParser, NCNlpServerManager, NCNlpWord}
import org.apache.nlpcraft.server.opencensus.NCOpenCensusServerStats

import scala.collection.JavaConverters._
import scala.collection.Map
import scala.util.control.Exception.catching

/**
  * TODO:
  */
object NCContextWordManager extends NCService with NCOpenCensusServerStats with NCIgniteInstance {
    private object Config extends NCConfigurable {
        lazy val url: Option[String] = getStringOpt("nlpcraft.server.ctxword.url")
    }

    private final val GSON = new Gson
    private final val TYPE_RESP = new TypeToken[java.util.List[java.util.List[java.util.List[Any]]]]() {}.getType
    private final val CLIENT = HttpClients.createDefault

    // TODO:CONF_REQ_MIN_SCORE
    private final val CONF_REQ_LIMIT = 20
    private final val CONF_REQ_MIN_SCORE = 0.7

    @volatile private var url: Option[String] = _
    @volatile private var parser: NCNlpParser = _
    @volatile private var cache: IgniteCache[NCContextRequest, Seq[NCContextWord]] = _

    // TODO: do we need all fields?
    case class Suggestion(word: String, totalScore: Double, fastTextScore: Double, bertScore: Double)
    case class RestRequest(sentences: java.util.List[java.util.List[Any]], limit: Int, min_score: Double)

    private final val HANDLER: ResponseHandler[Seq[Seq[Suggestion]]] =
        (resp: HttpResponse) ⇒ {
            val code = resp.getStatusLine.getStatusCode
            val e = resp.getEntity

            val js = if (e != null) EntityUtils.toString(e) else null

            if (js == null)
                throw new RuntimeException(s"Unexpected empty response [code=$code]")

            code match {
                case 200 ⇒
                    val data: java.util.List[java.util.List[java.util.List[Any]]] = GSON.fromJson(js, TYPE_RESP)

                    data.asScala.map(p ⇒
                        if (p.isEmpty)
                            Seq.empty
                        else
                            // Skips header.
                            p.asScala.tail.map(p ⇒
                                Suggestion(
                                    word = p.get(0).asInstanceOf[String],
                                    totalScore = p.get(1).asInstanceOf[Double],
                                    fastTextScore = p.get(2).asInstanceOf[Double],
                                    bertScore = p.get(3).asInstanceOf[Double]
                                )
                            )
                    )

                case 400 ⇒ throw new RuntimeException(js)
                case _ ⇒ throw new RuntimeException(s"Unexpected response [code=$code, response=$js]")
            }
        }

    @throws[NCE]
    def suggest(reqs: Seq[NCContextRequest], minScore: Double, limit: Int): Seq[Seq[NCContextWord]] = {
        require(url.isDefined)

        val res = scala.collection.mutable.LinkedHashMap.empty[NCContextRequest, Seq[NCContextWord]] ++
            reqs.map(key ⇒ key → cache.get(key))

        val reqsSrv = res.filter { case (_, cachedWords) ⇒ cachedWords == null }.keys.toSeq

        if (reqsSrv.nonEmpty) {
            val reqsSrvNorm: Seq[(String, Seq[Any])] =
                reqsSrv.groupBy(_.sentence).
                map { case (txt, seq) ⇒ txt → (Seq(txt) ++ seq.map(_.index).sorted) }.toSeq

            require(reqsSrv.size == reqsSrvNorm.map(_._2.size - 1).sum)

            val post = new HttpPost(url.get + "suggestions")

            println("!!limit="+limit)
            println("!!minScore="+minScore)
            println("!!reqsSrvNorm="+GSON.toJson(reqsSrvNorm.map(_._2.asJava).asJava))

            post.setHeader("Content-Type", "application/json")
            post.setEntity(
                new StringEntity(
                    GSON.toJson(RestRequest(reqsSrvNorm.map(_._2.asJava).asJava, limit, minScore)),
                    "UTF-8"
                )
            )

            val resp: Seq[Seq[NCContextWord]] =
                try
                    CLIENT.execute(post, HANDLER).
                        map(_.map(p ⇒
                            NCContextWord(word = p.word, stem = NCNlpCoreManager.stemWord(p.word), score = p.totalScore))
                        )
                finally
                    post.releaseConnection()

            require(reqsSrv.size == resp.size)

            val reqsSrvDenorm =
                reqsSrvNorm.flatMap { case (txt, vals) ⇒ vals.tail.map(idx ⇒ NCContextRequest(txt, idx.asInstanceOf[Int])) }

            def sort(seq: Seq[NCContextRequest]): Seq[NCContextRequest] = seq.sortBy(p ⇒ (p.sentence, p.index))

            require(sort(reqsSrvDenorm) == sort(reqsSrv))

            resp.
                zip(reqsSrvDenorm).
                sortBy { case (_, req) ⇒
                    val i = reqsSrv.indexOf(req)

                    require(i >= 0)

                    i
                }.
                foreach { case (seq, req) ⇒
                    cache.put(req, seq)
                    res.update(req, seq)
                }
        }

        val resVals = res.values.toSeq

        logger.info(s"Request [req=${reqs.mkString(", ")}, response=${resVals.map(_.mkString("|")).mkString(", ")}")

        resVals
    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { _ ⇒
        catching(wrapIE) {
            cache = ignite.cache[NCContextRequest, Seq[NCContextWord]]("ctxword-cache")
        }

        parser = NCNlpServerManager.getParser
        url = Config.url match {
            case Some(u) ⇒ Some(if (u.last == '/') u else s"$u/")
            case None ⇒ None
        }

        super.start()
    }

    override def stop(parent: Span = null): Unit = startScopedSpan("stop", parent) { _ ⇒
        cache = null

        super.stop()
    }

    private def substitute(words: Seq[String], subst: Map[Int, String]): String = {
        require(words.size >= subst.size)
        require(subst.keys.forall(i ⇒ i >= 0 && i < words.length))

        words.zipWithIndex.map { case (w, i) ⇒ subst.getOrElse(i, w) }.mkString(" ")
    }

    private def f(d: Double): String = "%1.3f" format d

    @throws[NCE]
    def makeContextWordConfig(
        mdlId: String,
        ctxSyns: Map[String /*Element ID*/ , Map[String /*Value*/, Set[String] /*Values synonyms stems*/ ]],
        examples: Set[String]
    ): NCContextWordConfigMdo = {
        val synonyms =
            ctxSyns.map { case (elemId, map) ⇒ elemId → map.flatMap { case (value, syns) ⇒ syns.map(_ → value) } }

        val contextWords =
            collection.mutable.HashMap.empty[String /*Element ID*/ , Map[String /*Context word stem*/ , Double /*Score*/]]

        val examplesCfg =
            collection.mutable.HashMap.empty[
                String /*Element ID*/ , Map[Seq[String] /*Synonyms tokens*/ , Seq[Int] /*Positions to substitute*/]
            ]

        val normExs: Set[Seq[NCNlpWord]] = examples.map(parser.parse(_))

        case class Holder(request: NCContextRequest, elementId: String, factor: Int)

        val allReqs: Seq[Holder] =
            ctxSyns.map { case (elemId, map) ⇒ elemId → map.values.flatten.toSet }.flatMap { case (elemId, allElemSyns) ⇒
                val elemNormExs: Map[Seq[NCNlpWord], Seq[Int]] =
                    normExs.flatMap(e ⇒ {
                        val indexes = e.zipWithIndex.flatMap {
                            case (w, idx) ⇒ if (allElemSyns.contains(w.stem)) Some(idx) else None
                        }

                        if (indexes.nonEmpty) Some(e → indexes) else None
                    }).toMap

                if (elemNormExs.isEmpty)
                    throw new NCE(s"Examples not found for element: $elemId")

                examplesCfg += elemId → elemNormExs.map { case (ex, idxs) ⇒ ex.map(_.word) → idxs }.toMap

                val n = elemNormExs.size * allElemSyns.size

                elemNormExs.flatMap { case (ex, idxs) ⇒
                    val exWords = ex.map(_.word)

                    allElemSyns.toSeq.combinations(idxs.size).flatMap(comb ⇒ {
                        require(comb.size == idxs.size)

                        val txt = substitute(exWords, idxs.zip(comb).toMap)

                        idxs.map(NCContextRequest(txt, _))
                    })
                }.toSeq.map(req ⇒ Holder(req, elemId, n))
            }.toSeq

        val allResp = suggest(allReqs.map(_.request), CONF_REQ_MIN_SCORE, CONF_REQ_LIMIT)

        require(allResp.size == allReqs.size)

        allReqs.zip(allResp).foreach { case (h, suggs) ⇒
            val suggsSumScores =
                suggs.
                    //filter(_.word.forall(ch ⇒ ch.isLetter && ch.isLower)). // TODO:
                    groupBy(_.stem).
                    filter(_._2.size > h.factor / 3.0). // Drop rare variants. TODO:
                    map { case (_, seq) ⇒ seq.minBy(-_.score) → seq.map(_.score).sum}.
                    toSeq.
                    sortBy { case(_, sumScore) ⇒ -sumScore }

            val maxSum = suggsSumScores.map { case (_, score) ⇒ score }.sum * 0.5

            val sum = new AtomicDouble(0.0)

            val suggsNorm = for ((s, sumScore) ← suggsSumScores if sum.getAndAdd(sumScore) < maxSum) yield s

            if (suggsNorm.isEmpty)
                throw new NCE(s"Context words cannot be prepared for element: ${h.elementId}")

            contextWords += h.elementId → suggsNorm.sortBy(-_.score).map(p ⇒ p.stem → p.score).toMap
        }

        logger.whenInfoEnabled({
            val tbl = NCAsciiTable()

            tbl #= ("Context word", "Score")

            contextWords.foreach { case (elemId, map) ⇒
                tbl += (s"Element ID: '$elemId'", "")

                map.toSeq.sortBy(-_._2).foreach { case (m, score) ⇒ tbl += (m, f(score)) }
            }

            tbl.info(logger, Some(s"Context words for model: $mdlId"))
        })

        NCContextWordConfigMdo(
            synonyms.toMap.map(p ⇒ p._1 → p._2.toMap),
            contextWords.toMap.map(p ⇒ p._1 → p._2.toMap),
            examplesCfg.toMap.map(p ⇒ p._1 → p._2.toMap)
        )
    }
}
