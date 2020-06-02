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

import java.util

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import io.opencensus.trace.Span
import org.apache.http.HttpResponse
import org.apache.http.client.ResponseHandler
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.HttpClients
import org.apache.http.util.EntityUtils
import org.apache.nlpcraft.common.config.NCConfigurable
import org.apache.nlpcraft.common.util.NCUtils
import org.apache.nlpcraft.common.{NCE, NCService}
import org.apache.nlpcraft.server.mdo.{NCElementSynonymMlMdo, NCModelMlConfigMdo}
import org.apache.nlpcraft.server.nlp.core.{NCNlpParser, NCNlpServerManager, NCNlpWord}
import org.apache.nlpcraft.server.opencensus.NCOpenCensusServerStats

import scala.collection.JavaConverters._

/**
  * TODO:
  */
object NCMlManager extends NCService with NCOpenCensusServerStats {
    private object Config extends NCConfigurable {
        lazy val url: Option[String] = getStringOpt("nlpcraft.server.ml.url")
    }

    case class RestRequest(sentence: String, simple: Boolean, lower: Int, upper: Int, limit: Int = 10)
    case class RestResponse(data: java.util.ArrayList[NCMlSuggestion])

    private final val GSON = new Gson
    private final val TYPE_RESP = new TypeToken[RestResponse]() {}.getType
    private final val CLIENT = HttpClients.createDefault

    @volatile private var url: Option[String] = _
    @volatile private var parser: NCNlpParser = _

    private case class Key(txt: String, idx: Int)

    private final val CACHE: util.Map[Key, Seq[NCMlSuggestion]] = NCUtils.mkLRUMap[Key, Seq[NCMlSuggestion]]("ml-cache", 10000)

    @throws[NCE]
    private def mkHandler(req: String): ResponseHandler[Seq[NCMlSuggestion]] =
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
    def ask(sen: String, idx: Int): Seq[NCMlSuggestion] = {
        require(url.isDefined)

        val key = Key(sen, idx)

        // TODO:
        CACHE.clear()

        var res = CACHE.synchronized { CACHE.get(key) }

        if (res != null)
            res
        else {
            val post = new HttpPost(url.get + "/synonyms")

            post.setHeader("Content-Type", "application/json")
            post.setEntity(
                new StringEntity(
                    GSON.toJson(
                        RestRequest(
                            sentence = sen,
                            simple = false,
                            lower = idx,
                            upper = idx
                        )
                    ),
                    "UTF-8"
                )
            )

            res =
                try
                    CLIENT.execute(post, mkHandler(sen))
                finally
                    post.releaseConnection()

            CACHE.synchronized { CACHE.put(key, res) }

            res
        }

    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { _ ⇒
        parser = NCNlpServerManager.getParser
        url = Config.url

        super.start()
    }

    override def stop(parent: Span = null): Unit = startScopedSpan("stop", parent) { _ ⇒
        CACHE.clear()

        super.stop()
    }

    @throws[NCE]
    def makeModelConfig(mlElems: Map[String, Set[String]], examples: Set[String]): NCModelMlConfigMdo = {
        val parsedExamples: Set[Seq[NCNlpWord]] = examples.map(parser.parse(_))

        val examplesCfg = scala.collection.mutable.HashMap.empty[String, Map[Seq[String], Int]]

        val mlElementsData =
            mlElems.map { case (elemId, synsStems) ⇒
                val elemExamples = parsedExamples.filter(_.exists(x ⇒ synsStems.contains(x.stem)))

                if (elemExamples.isEmpty)
                    throw new NCE(s"Examples not found for element: $elemId")

                case class Holder(synomym: NCElementSynonymMlMdo, words: Seq[String], index: Int)

                val hs =
                    elemExamples.flatMap(elemExample ⇒ {
                        val words = elemExample.map(_.word)
                        val normTxt = elemExample.map(_.normalWord).mkString(" ")

                        elemExample.
                            filter(e ⇒ synsStems.contains(e.stem)).
                            flatMap(n ⇒ {
                                val i = elemExample.indexOf(n)
                                val suggs = ask(normTxt, i)

                                suggs.map(s ⇒ Holder(NCElementSynonymMlMdo(s.word, s.score),words, i))
                            })
                    }).
                        //filter(_.synomym.word.forall(_.isLower)). // TODO: nouns
                        toSeq.sortBy(-_.synomym.score).
                        take(5) // TODO: 5,

                examplesCfg += elemId → hs.map(h ⇒ h.words → h.index).toMap

                elemId → hs.map(_.synomym)
            }

        val cfg = NCModelMlConfigMdo(mlElementsData, examplesCfg.toMap)

        logger.info(s"Config loaded: $cfg")

        cfg
    }
}
