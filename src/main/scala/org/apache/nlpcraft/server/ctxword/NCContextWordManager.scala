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
    private final val TYPE_RESP = new TypeToken[RestResponse]() {}.getType
    private final val CLIENT = HttpClients.createDefault

    // TODO:CONF_REQ_MIN_SCORE
    private final val CONF_REQ_LIMIT = 20
    private final val CONF_REQ_MIN_SCORE = 0.7

    @volatile private var url: Option[String] = _
    @volatile private var parser: NCNlpParser = _
    @volatile private var cache: IgniteCache[(String, Int), Seq[NCContextWord]] = _

    case class RestRequest(sentence: String, simple: Boolean, lower: Int, upper: Int, limit: Int)
    case class RestResponseBody(word: String, score: Double)
    case class RestResponse(data: java.util.ArrayList[RestResponseBody])

    @throws[NCE]
    private def mkHandler(req: String): ResponseHandler[Seq[RestResponseBody]] =
        (resp: HttpResponse) ⇒ {
            val code = resp.getStatusLine.getStatusCode
            val e = resp.getEntity

            val js = if (e != null) EntityUtils.toString(e) else null

            if (js == null)
                throw new NCE(s"Unexpected empty response [req=$req, code=$code]")

            code match {
                case 200 ⇒
                    val data: RestResponse = GSON.fromJson(js, TYPE_RESP)

                    data.data.asScala

                case 400 ⇒ throw new NCE(js)
                case _ ⇒ throw new NCE(s"Unexpected response [req=$req, code=$code, response=$js]")
            }
        }

    @throws[NCE]
    def suggest(sen: String, idx: Int, minScore: Double, limit: Int): Seq[NCContextWord] = {
        require(url.isDefined)

        val key: (String, Int) = (sen, idx)

        var res = cache.get(key)

        if (res == null) {
            val post = new HttpPost(url.get + "/synonyms")

            post.setHeader("Content-Type", "application/json")
            post.setEntity(
                new StringEntity(
                    GSON.toJson(RestRequest(sentence = sen, simple = false, lower = idx, upper = idx, limit = limit)),
                    "UTF-8"
                )
            )

            val reqRes =
                try
                    CLIENT.execute(post, mkHandler(sen)).filter(_.score >= minScore).sortBy(-_.score)
                finally
                    post.releaseConnection()

            res = reqRes.map(p ⇒ NCContextWord(word = p.word, stem = NCNlpCoreManager.stemWord(p.word), score = p.score))

            cache.put(key, res)
        }

//        logger.info(
//            s"Request sent [text=$sen, idx=$idx, result=${res.map(p ⇒ s"${p.word}(${f(p.score)})").mkString(", ")}]"
//        )
//
        res
    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { _ ⇒
        catching(wrapIE) {
            cache = ignite.cache[(String, Int), Seq[NCContextWord]]("ctxword-cache")
        }

        parser = NCNlpServerManager.getParser
        url = Config.url

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

        ctxSyns.map { case (elemId, map) ⇒ elemId → map.values.flatten.toSet }.foreach { case (elemId, allElemSyns) ⇒
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

            val suggsSumScores =
                elemNormExs.flatMap { case (ex, idxs) ⇒
                    val exWords = ex.map(_.word)

                    allElemSyns.toSeq.combinations(idxs.size).flatMap(comb ⇒ {
                        require(comb.size == idxs.size)

                        val txt = substitute(exWords, idxs.zip(comb).toMap)

                        idxs.flatMap(i ⇒ suggest(txt, i, CONF_REQ_MIN_SCORE, CONF_REQ_LIMIT))
                    })
                }.
                    toSeq.
                    //filter(_.word.forall(ch ⇒ ch.isLetter && ch.isLower)). // TODO:
                    groupBy(_.stem).
                    filter(_._2.size > n / 3.0). // Drop rare variants. TODO:
                    map { case (_, seq) ⇒ seq.sortBy(-_.score).head → seq.map(_.score).sum}.
                    toSeq.
                    sortBy { case(s, sumScore) ⇒ -sumScore}

            val suggs = collection.mutable.ArrayBuffer.empty[NCContextWord]

            val maxSum = suggsSumScores.map(_._2).sum * 0.5
            var sum = 0.0

            for ((s, sumScore) ← suggsSumScores if sum < maxSum) {
                suggs += s

                println("sum="+sum + ", s="+s)

                sum += sumScore
            }

            if (suggs.isEmpty)
                throw new NCE(s"Context words cannot be prepared for element: $elemId")

            contextWords += elemId → suggs.sortBy(-_.score).map(p ⇒ p.stem → p.score).toMap
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
