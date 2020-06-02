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

package org.apache.nlpcraft.server.nlp.enrichers.ml

import io.opencensus.trace.Span
import org.apache.nlpcraft.common.NCService
import org.apache.nlpcraft.common.config.NCConfigurable
import org.apache.nlpcraft.common.nlp.{NCNlpSentence, NCNlpSentenceNote}
import org.apache.nlpcraft.server.mdo.NCProbableSynonymMdo
import org.apache.nlpcraft.server.ml.NCMlManager
import org.apache.nlpcraft.server.nlp.enrichers.NCServerEnricher

object NCMlEnricher extends NCServerEnricher {
    @volatile private var url: String = _

    private object Config extends NCConfigurable {
        lazy val url: String = getStringOrElse("nlpcraft.server.ml.url", "http://localhost:5000")
    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { span ⇒
        url = Config.url

        if (url.last == '/')
            url = url.dropRight(1)

        addTags(span, "url" → url)

        // Tries to access spaCy proxy server.
        // TODO: add health check.

        logger.info(s"ML server connected: $url")

        super.start()
    }

    override def stop(parent: Span): Unit = startScopedSpan("stop", parent) { _ ⇒
        super.stop()
    }

    private def substitute(words: Seq[String], idx: Int, repl: String): String =
        words.zipWithIndex.map { case (w, i) ⇒ if (idx == i) repl else w }.mkString(" ")

    private def intersect(suggs: Seq[NCProbableSynonymMdo], syns: Map[NCProbableSynonymMdo, Boolean]): Option[Boolean] =
        suggs.flatMap(sug ⇒
            syns.flatMap {
                case (syn, isVal) ⇒
                    if (syn.word == sug.word) Some(isVal) else None
            }.toStream.headOption
        ).toStream.headOption


    override def enrich(ns: NCNlpSentence, parent: Span): Unit = {
        ns.mlCfg match {
            case Some(cfg) ⇒
                // TODO: other names.
                val nn = ns.filter(_.pos.startsWith("N"))

                if (nn.nonEmpty) {
                    val normTxt = ns.map(_.origText).mkString(" ")

                    nn.foreach(n ⇒ {
                        val idx = ns.indexOf(n)

                        val sugg = NCMlManager.ask(normTxt, idx, 0.5, 10) // TODO:

                        logger.info(s"Suggestions for main sentence [text=$normTxt, nn=${n.origText}, suggestions=${sugg.mkString(",")}]")

                        cfg.mlElements.flatMap { case (elemId, syns) ⇒
                            intersect(sugg, syns) match {
                                case Some(isVal) ⇒ Some(elemId, isVal)
                                case None ⇒ None
                            }
                        }.toStream.headOption match {
                            case Some((elemId, isVal)) ⇒
                                val tok = ns(idx)

                                // TODO: value,
                                tok.add(NCNlpSentenceNote(Seq(tok.index), elemId))


                            case None ⇒
                                cfg.examples.foreach { case (elemId, elemExamples) ⇒
                                    println("elemId="+elemId)
                                    println("elemExamples="+elemExamples)
                                    val all =
                                        elemExamples.forall { case (example, idx) ⇒
                                            val subs = substitute(example, idx, n.origText)
                                            val suggs = NCMlManager.ask(subs, idx, 0.5, 10)

                                            // TODO: value
                                            val ok = intersect(suggs, cfg.mlElements(elemId)).getOrElse(false)

                                            logger.info(s"Suggestions for examples [subs=$subs, i=$idx, nn=${n.origText}, suggestions=${suggs.mkString(",")}, ok=$ok]")

                                            ok
                                        }

                                    if (all) {
                                        val tok = ns(idx)

                                        tok.add(NCNlpSentenceNote(Seq(tok.index), elemId))
                                    }
                                }
                        }
                    })
                }
            case None ⇒ // No-op.
        }
    }
}
