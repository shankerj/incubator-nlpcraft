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

package org.apache.nlpcraft.server.ml

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
import org.apache.nlpcraft.server.mdo.{NCMlConfigMdo, NCMlSynonymInfoMdo}
import org.apache.nlpcraft.server.nlp.core.{NCNlpParser, NCNlpServerManager, NCNlpWord}
import org.apache.nlpcraft.server.opencensus.NCOpenCensusServerStats

import scala.collection.JavaConverters._
import scala.collection.Map
import scala.util.control.Exception.catching

/**
  * TODO:
  */
object NCMlManager extends NCService with NCOpenCensusServerStats with NCIgniteInstance {
    private object Config extends NCConfigurable {
        lazy val url: Option[String] = getStringOpt("nlpcraft.server.ml.url")
    }

    private final val GSON = new Gson
    private final val TYPE_RESP = new TypeToken[RestResponse]() {}.getType
    private final val CLIENT = HttpClients.createDefault

    // TODO:
    private final val CONF_COUNT_PER_EXAMPLE = 20
    private final val CONF_COUNT_SUM = 10
    private final val CONF_MIN_SCORE = 1

    @volatile private var url: Option[String] = _
    @volatile private var parser: NCNlpParser = _
    @volatile private var cache: IgniteCache[(String, Int), Seq[NCMLSuggestion]] = _

    case class RestRequest(sentence: String, simple: Boolean, lower: Int, upper: Int, limit: Int)
    case class RestResponseBody(word: String,score: Double)
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
    def suggest(sen: String, idx: Int, minScore: Double, limit: Int): Seq[NCMLSuggestion] = {
        require(url.isDefined)

        val key: (String, Int) = (sen, idx)

        var res = cache.get(key)

        // TODO:
        res = null

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

            res = reqRes.map(p ⇒ NCMLSuggestion(word = p.word, stem = NCNlpCoreManager.stemWord(p.word), score = p.score))

            cache.put(key, res)
        }

        logger.info(
            s"Request sent [text=$sen, idx=$idx, result=${res.map(p ⇒ s"${p.word}(${f(p.score)})").mkString(", ")}]"
        )

        res
    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { _ ⇒
        catching(wrapIE) {
            cache = ignite.cache[(String, Int), Seq[NCMLSuggestion]]("ml-cache")
        }

        parser = NCNlpServerManager.getParser
        url = Config.url

        super.start()
    }

    override def stop(parent: Span = null): Unit = startScopedSpan("stop", parent) { _ ⇒
        cache = null

        super.stop()
    }

    private def substitute(words: Seq[String], idx: Int, repl: String): String =
        words.zipWithIndex.map { case (w, i) ⇒ if (idx == i) repl else w }.mkString(" ")

    private def f(d: Double): String = "%1.3f" format d

    @throws[NCE]
    def makeMlConfig(
        mdlId: String,
        mlSyns: Map[String /*Element ID*/, Map[String /*Value*/, Set[String] /*Values synonyms stems*/]],
        examples: Set[String]
    ): NCMlConfigMdo = {
        val normExs: Set[Seq[NCNlpWord]] = examples.map(parser.parse(_))

        val examplesCfg =
            collection.mutable.HashMap.empty[
                String /*Element ID*/,
                Map[Seq[String] /*Synonyms tokens*/, Int /*Position to substitute*/]
            ].withDefault(_ ⇒ collection.mutable.HashMap.empty[Seq[String], Int])

        val synonyms  =
            collection.mutable.HashMap.empty[
                String /*Element ID*/, Map[String /*Synonym stem*/, NCMlSynonymInfoMdo /*Synonym info*/]
            ].withDefault(_ ⇒ collection.mutable.HashMap.empty[String, NCMlSynonymInfoMdo])

        case class Holder(info: NCMlSynonymInfoMdo, suggestionStem: String, words: Seq[String], index: Int)

        mlSyns.foreach { case (elemId, elemValsSyns) ⇒
            val elemNormExs = normExs.filter(_.exists(e ⇒ elemValsSyns.values.flatten.toSet.contains(e.stem)))

            if (elemNormExs.isEmpty)
                throw new NCE(s"Examples not found for element: $elemId")

            val hs =
                elemNormExs.flatMap(ex ⇒ {
                    val wordsEx = ex.map(_.word)

                    ex.flatMap(exWord ⇒
                        if (elemValsSyns.exists(_._2.contains(exWord.stem))) {
                            val i = ex.indexOf(exWord)

                            elemValsSyns.flatMap(p ⇒ p._2.map(_ → p._1)).flatMap {
                                case (syn, value) ⇒
                                        suggest(substitute(wordsEx, i, syn), i, 0, CONF_COUNT_PER_EXAMPLE).
                                            map(s ⇒ Holder(NCMlSynonymInfoMdo(s.score, value), s.stem, wordsEx, i))
                            }
                        }
                        else
                            Seq.empty
                    )
                })

            examplesCfg += elemId → hs.map(h ⇒ h.words → h.index).toMap

            hs.toSeq.map(h ⇒ h → NCNlpCoreManager.stemWord(h.suggestionStem)).
                groupBy { case (_, stem) ⇒ stem }.
                map { case (stem, seq) ⇒ stem → {
                    val sorted = seq.map { case (h, _) ⇒ h }.sortBy(-_.info.score)

                    val h = sorted.head

                    // TODO:
                    val score = sorted.map(_.info.score).sum

                    Holder(NCMlSynonymInfoMdo(score, h.info.value), h.suggestionStem, h.words, h.index)
                } }.
                toSeq.
                filter { case (_, h) ⇒ h.info.score >= CONF_MIN_SCORE }.
                sortBy { case (_, h) ⇒ -h.info.score }.
                take(CONF_COUNT_SUM).
                foreach { case (stem, h) ⇒ synonyms(elemId) += stem → h.info }
        }

        logger.whenInfoEnabled({
            logger.info(s"Model ML config: $mdlId")

            val tblSyns = NCAsciiTable()

            tblSyns #= ("Synonym", "Value", "Score")

            synonyms.foreach { case (elemId, map) ⇒
                tblSyns += (s"Element ID: '$elemId'", "", "")

                map.toSeq.sortBy(-_._2.score).foreach {
                    case (syn, info) ⇒ tblSyns += (syn, info.value, f(info.score))
                }
            }

            tblSyns.info(logger, Some("Synonyms:"))

            val tblEx = NCAsciiTable()

            tblEx #= ("Example", "Substitution index")

            examplesCfg.foreach { case (elemId, map) ⇒
                tblEx += (s"Element ID: '$elemId'", "")

                map.foreach { case (syn, idx) ⇒ tblEx += (syn.mkString(" "), idx)}
            }

            tblEx.info(logger, Some("Examples:"))
        })

        NCMlConfigMdo(synonyms.toMap.map(p ⇒ p._1 → p._2.toMap), examplesCfg.toMap.map(p ⇒ p._1 → p._2.toMap))
    }
}
