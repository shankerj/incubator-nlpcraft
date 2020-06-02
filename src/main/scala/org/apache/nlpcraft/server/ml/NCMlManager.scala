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
import org.apache.nlpcraft.common.config.NCConfigurable
import org.apache.nlpcraft.common.{NCE, NCService}
import org.apache.nlpcraft.server.ignite.NCIgniteInstance
import org.apache.nlpcraft.server.mdo.{NCProbableSynonymMdo, NCMlConfigMdo}
import org.apache.nlpcraft.server.nlp.core.{NCNlpParser, NCNlpServerManager, NCNlpWord}
import org.apache.nlpcraft.server.opencensus.NCOpenCensusServerStats

import scala.collection.JavaConverters._
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

    @volatile private var url: Option[String] = _
    @volatile private var parser: NCNlpParser = _
    @volatile private var cache: IgniteCache[(String, Int), Seq[NCProbableSynonymMdo]] = _

    case class RestRequest(sentence: String, simple: Boolean, lower: Int, upper: Int, limit: Int)
    case class RestResponse(data: java.util.ArrayList[NCProbableSynonymMdo])

    @throws[NCE]
    private def mkHandler(req: String): ResponseHandler[Seq[NCProbableSynonymMdo]] =
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
    def ask(sen: String, idx: Int, minScore: Double, limit: Int): Seq[NCProbableSynonymMdo] = {
        require(url.isDefined)

        val key: (String, Int) = (sen, idx)

        var res = cache.get(key)

        if (res != null)
            res
        else {
            val post = new HttpPost(url.get + "/synonyms")

            post.setHeader("Content-Type", "application/json")
            post.setEntity(
                new StringEntity(
                    GSON.toJson(RestRequest(sentence = sen, simple = false, lower = idx, upper = idx, limit = limit)),
                    "UTF-8"
                )
            )

            res =
                try
                    CLIENT.execute(post, mkHandler(sen)).filter(_.score >= minScore).sortBy(-_.score)
                finally
                    post.releaseConnection()

            cache.put(key, res)

            res
        }
    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { _ ⇒
        catching(wrapIE) {
            cache = ignite.cache[(String, Int), Seq[NCProbableSynonymMdo]]("ml-cache")
        }

        parser = NCNlpServerManager.getParser
        url = Config.url

        super.start()
    }

    override def stop(parent: Span = null): Unit = startScopedSpan("stop", parent) { _ ⇒
        cache = null

        super.stop()
    }

    @throws[NCE]
    def makeModelConfig(mlElems: Map[String, Map[String, Boolean]], examples: Set[String]): NCMlConfigMdo = {
        val parsedExamples: Set[Seq[NCNlpWord]] = examples.map(parser.parse(_))

        val examplesCfg = scala.collection.mutable.HashMap.empty[String, Map[Seq[String], Int]]

        val mlElementsData =
            mlElems.map { case (elemId, syns) ⇒
                val elemExamples = parsedExamples.filter(_.exists(e ⇒ syns.keySet.contains(e.stem)))

                if (elemExamples.isEmpty)
                    throw new NCE(s"Examples not found for element: $elemId")

                case class Holder(synomym: NCProbableSynonymMdo, isValue: Boolean, words: Seq[String], index: Int)

                val hs =
                    elemExamples.flatMap(elemExample ⇒ {
                        val words = elemExample.map(_.word)
                        val normTxt = elemExample.map(_.normalWord).mkString(" ")

                        elemExample.
                            flatMap(word ⇒
                                syns.get(word.normalWord) match {
                                    case Some(isValue) ⇒
                                        val i = elemExample.indexOf(word)
                                        val suggs = ask(normTxt, i, 0, 5) // TODO:

                                        suggs.map(s ⇒ Holder(NCProbableSynonymMdo(s.word, s.score), isValue, words, i))
                                    case None ⇒ Seq.empty
                                }
                            )
                    })
                        //filter(_.synomym.word.forall(_.isLower)). // TODO: nouns

                examplesCfg += elemId → hs.map(h ⇒ h.words → h.index).toMap

                elemId → hs.map(h ⇒ h.synomym → h.isValue)
            }.toMap

        val cfg = NCMlConfigMdo(mlElementsData, examplesCfg.toMap)

        logger.info(s"Config loaded: $cfg")

        cfg
    }
}
