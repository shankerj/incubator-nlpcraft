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

import java.net.ConnectException
import java.util.{List ⇒ JList}

import com.google.gson.reflect.TypeToken
import com.google.gson.{Gson, GsonBuilder}
import io.opencensus.trace.Span
import org.apache.http.HttpResponse
import org.apache.http.client.ResponseHandler
import org.apache.http.client.methods.{HttpGet, HttpPost}
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
import scala.util.control.Exception.catching

/**
  * TODO:
  */
object NCContextWordManager extends NCService with NCOpenCensusServerStats with NCIgniteInstance {
    private object Config extends NCConfigurable {
        lazy val url: Option[String] = getStringOpt("nlpcraft.server.ctxword.url")
    }

    private final val GSON = new Gson()
    private final val TYPE_RESP = new TypeToken[JList[JList[Suggestion]]]() {}.getType
    private final val CLIENT = HttpClients.createDefault

    private final val CONF_LIMIT = 20
    private final val CONF_MIN_SCORE = 0.7

    @volatile private var url: Option[String] = _
    @volatile private var parser: NCNlpParser = _
    @volatile private var cache: IgniteCache[NCContextWordRequest, Seq[NCContextWordResponse]] = _

    // We don't use directly bert and ftext indexes.
    case class Suggestion(word: String, score: Double, bert_score: Double, ftext_score: Double)
    case class RestSentence(text: String, indexes: JList[Int])
    case class RestRequest(
        sentences: JList[RestSentence],
        limit: Int,
        min_score: Double,
        min_ftext: Double,
        min_bert: Double
    )

    private final val HANDLER: ResponseHandler[Seq[Seq[Suggestion]]] =
        (resp: HttpResponse) ⇒ {
            val code = resp.getStatusLine.getStatusCode
            val e = resp.getEntity

            val js = if (e != null) EntityUtils.toString(e) else null

            if (js == null)
                throw new RuntimeException(s"Unexpected empty response [code=$code]")

            code match {
                case 200 ⇒
                    val data: JList[JList[Suggestion]] = GSON.fromJson(js, TYPE_RESP)

                    // Skips header.
                    data.asScala.map(p ⇒ if (p.isEmpty) Seq.empty else p.asScala.tail)

                case 400 ⇒ throw new RuntimeException(js)
                case _ ⇒ throw new RuntimeException(s"Unexpected response [code=$code, response=$js]")
            }
        }

    @throws[NCE]
    def suggest(reqs: Seq[NCContextWordRequest], f: NCContextWordParameter): Seq[Seq[NCContextWordResponse]] = {
        require(url.isDefined)

        val res = scala.collection.mutable.LinkedHashMap.empty[NCContextWordRequest, Seq[NCContextWordResponse]] ++
            reqs.map(key ⇒ key → cache.get(key))

        val reqsSrv = res.filter { case (_, cachedWords) ⇒ cachedWords == null }.keys.toSeq

        if (reqsSrv.nonEmpty) {
            val reqsSrvNorm: Seq[(Seq[String], RestSentence)] =
                reqsSrv.groupBy(_.words).
                map {
                    case (words, seq) ⇒ words → RestSentence(words.mkString(" "), seq.map(_.wordIndex).sorted.asJava)
                }.toSeq

            require(reqsSrv.size == reqsSrvNorm.map(_._2.indexes.size).sum)

            val restReq =
                RestRequest(
                    sentences = reqsSrvNorm.map { case (_, restSen) ⇒ restSen }.asJava,
                    limit = f.limit,
                    min_score = f.totalScore,
                    min_ftext = f.ftextScore,
                    min_bert = f.bertScore
                )

            val post = new HttpPost(url.get)

            post.setHeader("Content-Type", "application/json")
            post.setEntity(new StringEntity(GSON.toJson(restReq), "UTF-8"))

            val resp: Seq[Seq[NCContextWordResponse]] =
                try
                    CLIENT.execute(post, HANDLER).
                        map(_.map(p ⇒
                            NCContextWordResponse(
                                word = p.word,
                                stem = NCNlpCoreManager.stemWord(p.word),
                                totalScore = p.score,
                                bertScore = p.bert_score,
                                ftextScore = p.ftext_score
                            ))
                        )
                finally
                    post.releaseConnection()

            require(reqsSrv.size == resp.size)

            val reqsSrvDenorm =
                reqsSrvNorm.flatMap { case (txt, sen) ⇒ sen.indexes.asScala.map(NCContextWordRequest(txt, _)) }

            def sort(seq: Seq[NCContextWordRequest]): Seq[NCContextWordRequest] =
                seq.sortBy(p ⇒ (p.words.mkString(" "), p.wordIndex))

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

        // TODO: level
        logger.whenInfoEnabled({
            logger.info(s"Request executed: \n${
                new GsonBuilder().
                    setPrettyPrinting().
                    create.
                    toJson(
                        res.map { case (req, resp) ⇒ (req.words.mkString(" "), req.wordIndex) → resp.asJava } .asJava
                    )
            }")
        })

        res.values.toSeq
    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { _ ⇒
        catching(wrapIE) {
            cache = ignite.cache[NCContextWordRequest, Seq[NCContextWordResponse]]("ctxword-cache")
        }

        parser = NCNlpServerManager.getParser
        url = Config.url match {
            case Some(u) ⇒
                // It doesn't even check return code, just catch connection exception.
                try
                    HttpClients.createDefault.execute(new HttpGet(u))
                catch {
                    case e: ConnectException ⇒ throw new NCE(s"Service is not available: $u", e)
                }

                Some(if (u.last == '/') s"${u}suggestions" else s"$u/suggestions")
            case None ⇒ None
        }

        super.start()
    }

    override def stop(parent: Span = null): Unit = startScopedSpan("stop", parent) { _ ⇒
        cache = null

        super.stop()
    }

    private def substitute(words: Seq[String], subst: Map[Int, String]): Seq[String] = {
        require(words.size >= subst.size)
        require(subst.keys.forall(i ⇒ i >= 0 && i < words.length))

        words.zipWithIndex.map { case (w, i) ⇒ subst.getOrElse(i, w) }
    }

    @throws[NCE]
    def makeConfig(
        mdlId: String,
        ctxSyns: Map[String /*Element ID*/ , Map[String /*Value*/, Set[String] /*Values synonyms stems*/ ]],
        examples: Set[String]
    ): NCContextWordConfigMdo = {
        val synonyms =
            ctxSyns.map { case (elemId, map) ⇒ elemId → map.flatMap { case (value, syns) ⇒ syns.map(_ → value) } }

        val contextWords = collection.mutable.HashMap.empty[String /*Element ID*/, Set[String] /*Stems*/]

        val examplesCfg =
            collection.mutable.HashMap.empty[
                String /*Element ID*/ , Map[Seq[String] /*Synonyms tokens*/, Seq[Int] /*Positions to substitute*/]
            ]

        val normExs: Set[Seq[NCNlpWord]] = examples.map(parser.parse(_))

        case class Holder(request: NCContextWordRequest, elementId: String, factor: Int)

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

                examplesCfg += elemId → elemNormExs.map { case (ex, idxs) ⇒ ex.map(_.word) → idxs }

                val multFactor = elemNormExs.size * allElemSyns.size

                elemNormExs.flatMap { case (ex, idxs) ⇒
                    val exWords = ex.map(_.word)

                    allElemSyns.toSeq.combinations(idxs.size).flatMap(comb ⇒ {
                        require(comb.size == idxs.size)

                        val words = substitute(exWords, idxs.zip(comb).toMap)

                        idxs.map(NCContextWordRequest(words, _))
                    })
                }.toSeq.map(req ⇒ Holder(req, elemId, multFactor))
            }.toSeq

        val allResp = suggest(
            allReqs.map(_.request),
            NCContextWordParameter(limit = CONF_LIMIT, totalScore = CONF_MIN_SCORE)
        )

        require(allResp.size == allReqs.size)

        val groupsByElem = allReqs.zip(allResp).groupBy { case (h, _) ⇒ (h.elementId, h.factor) }

        require(groupsByElem.size == ctxSyns.size)

        groupsByElem.foreach { case ((elemId, factor), seq) ⇒
            val suggs = seq.flatMap { case (_, seq) ⇒ seq }

            if (suggs.isEmpty)
                throw new NCE(s"Context words cannot be prepared for element: '$elemId'")

            val top = suggs.groupBy(_.stem).map { case (_, group) ⇒
                val best = group.minBy(-_.totalScore)

                best → (group.size, best.totalScore)
                // TODO:
            }.toSeq.sortBy( p ⇒ (-p._2._1, -p._2._2)).take(10)

            // TODO:
            println("TOP:" + top)

            contextWords += elemId → top.map(_._1).map(_.stem).toSet
        }

        logger.whenInfoEnabled({
            val tbl = NCAsciiTable()

            tbl #= "Context word"

            contextWords.foreach { case (elemId, set) ⇒
                tbl += s"Element ID: '$elemId'"

                set.foreach(s ⇒ tbl += s)
            }

            tbl.info(logger, Some(s"Context words for model: $mdlId"))
        })

        NCContextWordConfigMdo(synonyms, contextWords.toMap, examplesCfg.toMap)
    }
}
